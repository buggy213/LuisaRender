//
// Work Graph Path Tracing Integrator (Wavefront style)
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/thread_pool.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <dsl/syntax.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>
#include <luisa/backends/ext/work_graph_ext.h>
#include <luisa/runtime/work_graph/work_graph_program.h>

namespace luisa::render {

using namespace compute;

static constexpr uint WG_BLOCK_SIZE = 64u;

class PathStateSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _wl_sample;
    Buffer<float> _beta;
    // Buffer<float> _radiance;
    Buffer<float> _pdf_bsdf;
    Buffer<uint> _data; // samples << 8 | depth << 1 | extend_ray

public:
    PathStateSOA(const Spectrum::Instance *spectrum, size_t size) noexcept
        : _spectrum{spectrum} {
        auto &&device = spectrum->pipeline().device();
        auto dimension = spectrum->node()->dimension();
        _beta = device.create_buffer<float>(size * dimension);
        // _radiance = device.create_buffer<float>(size * dimension);
        _pdf_bsdf = device.create_buffer<float>(size);
        _data = device.create_buffer<uint>(size);
        if (!spectrum->node()->is_fixed()) {
            _wl_sample = device.create_buffer<float>(size);
        }
    }
    [[nodiscard]] auto read_beta(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _beta->read(offset + i);
        }
        return s;
    }
    void write_beta(Expr<uint> index, const SampledSpectrum &beta) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _beta->write(offset + i, beta[i]);
        }
    }
    [[nodiscard]] auto read_swl(Expr<uint> index) const noexcept {
        if (_spectrum->node()->is_fixed()) {
            return std::make_pair(def(0.f), _spectrum->sample(0.f));
        }
        auto u_wl = _wl_sample->read(index);
        auto swl = _spectrum->sample(abs(u_wl));
        $if(u_wl < 0.f) { swl.terminate_secondary(); };
        return std::make_pair(abs(u_wl), swl);
    }
    void write_wavelength_sample(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample->write(index, u_wl);
        }
    }
    void terminate_secondary_wavelengths(Expr<uint> index, Expr<float> u_wl) noexcept {
        if (!_spectrum->node()->is_fixed()) {
            _wl_sample->write(index, -u_wl);
        }
    }
    // [[nodiscard]] auto read_radiance(Expr<uint> index) const noexcept {
    //     auto dimension = _spectrum->node()->dimension();
    //     auto offset = index * dimension;
    //     SampledSpectrum s{dimension};
    //     for (auto i = 0u; i < dimension; i++) {
    //         s[i] = _radiance->read(offset + i);
    //     }
    //     return s;
    // }
    // void write_radiance(Expr<uint> index, const SampledSpectrum &s) noexcept {
    //     auto dimension = _spectrum->node()->dimension();
    //     auto offset = index * dimension;
    //     for (auto i = 0u; i < dimension; i++) {
    //         _radiance->write(offset + i, s[i]);
    //     }
    // }
    [[nodiscard]] auto read_pdf_bsdf(Expr<uint> index) const noexcept {
        return _pdf_bsdf->read(index);
    }
    void write_pdf_bsdf(Expr<uint> index, Expr<float> pdf) noexcept {
        _pdf_bsdf->write(index, pdf);
    }

    [[nodiscard]] auto read_data(Expr<uint> index) const noexcept {
        Var<uint> data = _data->read(index);
        Var<uint> samples = data >> 8;
        Var<uint> depth = (data >> 1) & 0b1111111;
        Var<uint> extend_ray = data & 1;
        return std::make_tuple(samples, depth, extend_ray);
    }

    void write_data(Expr<uint> index, Expr<uint> samples, Expr<uint> depth, Expr<uint> extend_ray) noexcept {
        Var<uint> data = samples << 8 | depth << 1 | extend_ray;
        _data->write(index, data);
    }
};

class LightSampleSOA {

private:
    const Spectrum::Instance *_spectrum;
    Buffer<float> _emission;
    Buffer<float4> _wi_and_pdf;

public:
    LightSampleSOA(const Spectrum::Instance *spec, size_t size) noexcept
        : _spectrum{spec} {
        auto &&device = spec->pipeline().device();
        auto dimension = spec->node()->dimension();
        _emission = device.create_buffer<float>(size * dimension);
        _wi_and_pdf = device.create_buffer<float4>(size);
    }
    [[nodiscard]] auto read_emission(Expr<uint> index) const noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        SampledSpectrum s{dimension};
        for (auto i = 0u; i < dimension; i++) {
            s[i] = _emission->read(offset + i);
        }
        return s;
    }
    void write_emission(Expr<uint> index, const SampledSpectrum &s) noexcept {
        auto dimension = _spectrum->node()->dimension();
        auto offset = index * dimension;
        for (auto i = 0u; i < dimension; i++) {
            _emission->write(offset + i, s[i]);
        }
    }
    [[nodiscard]] auto read_wi_and_pdf(Expr<uint> index) const noexcept {
        return _wi_and_pdf->read(index);
    }
    void write_wi_and_pdf(Expr<uint> index, Expr<float3> wi, Expr<float> pdf) noexcept {
        _wi_and_pdf->write(index, make_float4(wi, pdf));
    }
};


// ============================================================================
// Record Structs
// ============================================================================

struct IntersectRecord {
    uint pixel_id;
};

struct PostIntersectRecord {
    uint pixel_id;
};

struct SurfaceRecord {
    uint pixel_id;
};

// Entry record: dispatch grid size + per-dispatch constants
struct WGEntryRecord : DispatchGridRecord {};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::IntersectRecord, pixel_id) {};

LUISA_STRUCT(luisa::render::PostIntersectRecord, pixel_id) {};

LUISA_STRUCT(luisa::render::SurfaceRecord, pixel_id) {};

LUISA_STRUCT(luisa::render::WGEntryRecord, size) {};

namespace luisa::render {

// ============================================================================
// Integrator class (scene node)
// ============================================================================

class WorkGraphPathTracing final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    bool _unrolled;
    luisa::string _binning_mode;

public:
    WorkGraphPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 5u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 2u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _unrolled{desc->property_bool_or_default("unrolled", false)},
          _binning_mode{desc->property_string_or_default("material_binning", "single")} {}

    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto unrolled() const noexcept { return _unrolled; }
    [[nodiscard]] auto &binning_mode() const noexcept { return _binning_mode; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

// ============================================================================
// Integrator instance
// ============================================================================

class WorkGraphPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override;

private:
    [[nodiscard]] luisa::vector<uint> _compute_tag_to_group() const noexcept {
        auto num_surfaces = pipeline().surfaces().size();
        luisa::vector<uint> tag_to_group(num_surfaces);
        auto &mode = node<WorkGraphPathTracing>()->binning_mode();
        if (mode == "single") {
            for (auto i = 0u; i < num_surfaces; i++) tag_to_group[i] = 0u;
        } else {
            for (auto i = 0u; i < num_surfaces; i++) tag_to_group[i] = i;
        }
        return tag_to_group;
    }

    [[nodiscard]] static uint _num_groups(const luisa::vector<uint> &tag_to_group) noexcept {
        uint max_group = 0u;
        for (auto g : tag_to_group) max_group = std::max(max_group, g);
        return max_group + 1u;
    }

    [[nodiscard]] WorkGraph _build_multi_dispatch_graph(
        Camera::Instance *camera,
        const luisa::vector<uint> &tag_to_group,
        uint num_groups,
        uint2 resolution,
        PathStateSOA& path_states,
        LightSampleSOA& light_samples,
        Buffer<Ray>& rays,
        Buffer<Hit>& hits,
        Buffer<uint>& active_count,
        uint max_dispatch_groups) noexcept;
};

luisa::unique_ptr<Integrator::Instance> WorkGraphPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<WorkGraphPathTracingInstance>(pipeline, command_buffer, this);
}

// ============================================================================
// Work graph construction (multi-dispatch mode)
// ============================================================================

WorkGraph WorkGraphPathTracingInstance::_build_multi_dispatch_graph(
    Camera::Instance *camera,
    const luisa::vector<uint> &tag_to_group,
    uint num_groups,
    uint2 resolution,
    PathStateSOA& path_states,
    LightSampleSOA& light_samples,
    Buffer<Ray>& rays,
    Buffer<Hit>& hits,
    Buffer<uint>& active_count,
    uint max_dispatch_groups) noexcept {

    auto spectrum = pipeline().spectrum();
    auto dim = spectrum->node()->dimension();
    LUISA_ASSERT(dim <= 4u, "Spectrum dimension {} exceeds 4.", dim);

    auto max_depth = node<WorkGraphPathTracing>()->max_depth();
    auto rr_depth = node<WorkGraphPathTracing>()->rr_depth();
    auto rr_threshold = node<WorkGraphPathTracing>()->rr_threshold();
    auto spp = camera->node()->spp();

    bool has_environment_light = pipeline().environment() != nullptr;
    bool has_localized_light = !pipeline().lights().empty();

    WorkGraphBuilder builder{"wg-path-bounce"};

    // --- Entry node (BROADCASTING, dynamic dispatch grid) ---
    auto entry = builder.add_node<WorkGraphLaunchType::BROADCASTING, WGEntryRecord>("entry");
    entry.set_threadgroup_size({WG_BLOCK_SIZE, 1, 1});
    entry.set_max_dispatch_size({max_dispatch_groups, 1, 1});

    // MaxRecords is per thread group, not per thread.
    // Each of the WG_BLOCK_SIZE threads may write to at most one of these outputs.
    luisa::optional<WorkGraphNodeOutput<PostIntersectRecord>> to_miss;
    if (has_environment_light) {
        to_miss = entry.output<PostIntersectRecord>(WG_BLOCK_SIZE);
    }

    luisa::optional<WorkGraphNodeOutput<PostIntersectRecord>> to_light;
    if (has_localized_light) {
        to_light = entry.output<PostIntersectRecord>(WG_BLOCK_SIZE);
    }

    auto to_light_sample = entry.output<PostIntersectRecord>(WG_BLOCK_SIZE);

    WorkGraphNodeKernel entry_kernel = [&](Var<WGEntryRecord> entry_rec) {
        auto thread_id = dispatch_x();
        auto [samples, depth, extend_ray] = path_states.read_data(thread_id);

        auto total_pixels = resolution.x * resolution.y;
        Var<PostIntersectRecord> post;
        Bool active = (thread_id < total_pixels) & samples != camera->node()->spp();
        Bool miss = def(false);
        Bool has_light = def(false);
        Bool has_surface = def(false);
        $if (active) {
            // active_count->atomic(0).fetch_add(1u);
            Var<Ray> ray;

            // Inline raygen
            $if (extend_ray == 0u) { 
                auto pixel_coord = make_uint2(thread_id % resolution.x,
                                              thread_id / resolution.x);
                sampler()->start(pixel_coord, samples);
                auto u_filter = sampler()->generate_pixel_2d();
                auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
                auto u_wl = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
                sampler()->save_state(thread_id);

                auto camera_sample = camera->generate_ray(pixel_coord, 0.0f, u_filter, u_lens);

                camera->film()->accumulate(pixel_coord, make_float3(0.f), 1.f);

                ray = camera_sample.ray;
                rays->write(thread_id, ray);
                path_states.write_data(
                    thread_id,
                    samples,
                    0u,
                    0u
                );
                path_states.write_pdf_bsdf(thread_id, 1e16f);
                path_states.write_wavelength_sample(thread_id, u_wl);
                path_states.write_beta(thread_id, SampledSpectrum{dim, camera_sample.weight});
            }
            $else {
                ray = rays->read(thread_id);
            };

            auto hit = pipeline().geometry()->trace_closest(ray);
            post.pixel_id = thread_id;

            miss = hit->miss();
            $if (!miss) {
                auto shape = pipeline().geometry()->instance(hit.inst);
                has_light = shape.has_light();
                has_surface = shape.has_surface();
                hits->write(thread_id, hit);
            };
        };

        if (has_environment_light) {
            to_miss->write(post, active & miss);
        }
        if (has_localized_light) {
            to_light->write(post, active & !miss & has_light);
        }

        to_light_sample.write(post, active & !miss & !has_light & has_surface);
    };
    entry.define(entry_kernel);

    // --- Miss node ---
    if (has_environment_light) {
        auto miss_node = builder.add_node<WorkGraphLaunchType::THREAD, PostIntersectRecord>("miss");

        WorkGraphNodeKernel miss_kernel = [&](Var<PostIntersectRecord> input) {
            Var<Ray> ray = rays->read(input.pixel_id);
            auto wi = ray->direction();
            auto [_, swl] = path_states.read_swl(input.pixel_id);
            auto beta = path_states.read_beta(input.pixel_id);
            Float pdf_bsdf = path_states.read_pdf_bsdf(input.pixel_id);
            auto eval = light_sampler()->evaluate_miss(wi, swl, 0.f);
            auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
            auto Li = beta * eval.L * mis_weight;
            auto pixel_coord = make_uint2(input.pixel_id % resolution.x,
                                          input.pixel_id / resolution.x);
            camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);

            // Path terminated, bump sample count and inform raygen
            auto [samples, depth, _extend_ray] = path_states.read_data(input.pixel_id);
            path_states.write_data(input.pixel_id, samples + 1, 0u, 0u);
        };
        miss_node.define(miss_kernel);
        miss_node << *to_miss;
    }

    // --- Light node ---
    luisa::optional<WorkGraphNode<WorkGraphLaunchType::THREAD, PostIntersectRecord>> light_node;
    luisa::optional<WorkGraphNodeOutput<PostIntersectRecord>> light_to_sample;

    if (has_localized_light) {
        light_node = builder.add_node<WorkGraphLaunchType::THREAD, PostIntersectRecord>("light");
        light_to_sample = light_node->output<PostIntersectRecord>(1);

        WorkGraphNodeKernel light_kernel = [&](Var<PostIntersectRecord> input) {
            Var<Ray> ray = rays->read(input.pixel_id);
            Var<Hit> hit = hits->read(input.pixel_id);
            auto [_, swl] = path_states.read_swl(input.pixel_id);
            auto beta = path_states.read_beta(input.pixel_id);
            Float pdf_bsdf = path_states.read_pdf_bsdf(input.pixel_id);
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, 0.f);
            auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
            auto Li = beta * eval.L * mis_weight;
            auto pixel_coord = make_uint2(input.pixel_id % resolution.x,
                                          input.pixel_id / resolution.x);
            camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);

            auto shape = pipeline().geometry()->instance(hit.inst);
            auto has_surf = shape.has_surface();
            // Path terminates here for pure emitters
            $if(!has_surf) {
                auto [samples, depth, _extend_ray] = path_states.read_data(input.pixel_id);
                path_states.write_data(input.pixel_id, samples + 1, 0u, 0u);
            };
            light_to_sample->write(input, has_surf);
        };
        light_node->define(light_kernel);
        *light_node << *to_light;
    }

    // --- LightSample node ---
    auto light_sample_node = builder.add_node<WorkGraphLaunchType::THREAD, PostIntersectRecord>("light_sample");
    auto to_surface = light_sample_node.array_output<SurfaceRecord>(1);

    WorkGraphNodeKernel light_sample_kernel = [&](Var<PostIntersectRecord> input) {
        auto pixel_id = input.pixel_id;
        sampler()->load_state(pixel_id);
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        sampler()->save_state(pixel_id);

        Var<Ray> ray = rays->read(pixel_id);
        Var<Hit> hit = hits->read(pixel_id);
        auto it = pipeline().geometry()->interaction(ray, hit);
        auto [_, swl] = path_states.read_swl(pixel_id);

        auto light_sample = light_sampler()->sample(*it, u_light_selection, u_light_surface, swl, 0.0f);

        Bool occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

        Var<SurfaceRecord> surf_rec;
        surf_rec.pixel_id = input.pixel_id;

        auto emission_spec = ite(occluded, SampledSpectrum{dim, 0.0f}, light_sample.eval.L);
        light_samples.write_emission(pixel_id, emission_spec);

        Float3 light_wi = light_sample.shadow_ray->direction();
        light_samples.write_wi_and_pdf(pixel_id, light_wi, ite(occluded, 0.0f, light_sample.eval.pdf));

        // Compile-time tag-to-group lookup
        auto group = def(0u);
        auto surface_tag = it->shape().surface_tag();
        $switch(surface_tag) {
            for (uint t = 0u; t < tag_to_group.size(); t++) {
                $case(t) { group = uint(tag_to_group[t]); };
            }
        };

        // Evaluate surface
        to_surface.write(surf_rec, group, true);
    };
    light_sample_node.define(light_sample_kernel);
    light_sample_node << to_light_sample;
    if (has_localized_light) {
        light_sample_node << *light_to_sample;
    }

    // --- Surface node array ---
    auto surface_nodes = builder.add_node_array<WorkGraphLaunchType::THREAD, SurfaceRecord>(
        "surface", num_groups);

    for (uint g = 0u; g < num_groups; g++) {
        WorkGraphNodeKernel surface_kernel = [&, g](Var<SurfaceRecord> input) {
            auto pixel_id = input.pixel_id;
            auto [samples, depth, _extend_ray] = path_states.read_data(pixel_id);
            sampler()->load_state(pixel_id);
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            $if(depth + 1u >= rr_depth) {
                u_rr = sampler()->generate_1d();
            };
            sampler()->save_state(pixel_id);

            Var<Ray> ray = rays->read(pixel_id);
            Var<Hit> hit = hits->read(pixel_id);
            auto it = pipeline().geometry()->interaction(ray, hit);

            auto [u_wl, swl] = path_states.read_swl(pixel_id);
            auto beta = path_states.read_beta(pixel_id);

            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            auto wo = -ray->direction();

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, 0.f);
            });

            auto new_ray = def(ray);
            auto new_pdf_bsdf = def(0.f);
            auto terminated = def(false);

            call.execute([&](const Surface::Closure *closure) noexcept {
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) {
                        swl.terminate_secondary();
                        path_states.terminate_secondary_wavelengths(pixel_id, u_wl);
                    };
                }

                // Direct lighting
                Float4 light_wi_and_pdf = light_samples.read_wi_and_pdf(pixel_id);
                auto light_wi = light_wi_and_pdf.xyz();
                auto pdf_light = light_wi_and_pdf.w;
                $if(pdf_light > 0.0f) {
                    auto eval = closure->evaluate(wo, light_wi);
                    auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                    auto Ld = light_samples.read_emission(pixel_id);
                    auto Li = mis_weight / pdf_light * beta * eval.f * Ld;
                    auto pixel_coord = make_uint2(pixel_id % resolution.x,
                                                  pixel_id / resolution.x);
                    camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
                };

                // Sample BSDF for next bounce
                auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                new_pdf_bsdf = surface_sample.eval.pdf;
                new_ray = it->spawn_ray(surface_sample.wi);
                auto w = ite(surface_sample.eval.pdf > 0.0f,
                             1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;

                auto eta = closure->eta().value_or(1.f);
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = 1.f / sqr(eta); };
                };
            });

            // Prepare for next bounce
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) {
                terminated = true;
            }
            $else {
                auto q = max(beta.max() * eta_scale, 0.05f);
                $if(depth + 1u >= rr_depth) {
                    terminated = q < rr_threshold & u_rr >= q;
                    beta *= ite(q < rr_threshold, 1.f / q, 1.f);
                };
            };

            $if(depth + 1u >= max_depth) {
                terminated = true;
            };

            $if(terminated) {
                path_states.write_data(pixel_id, samples + 1u, 0u, 0u);
            } $else {
                rays->write(pixel_id, new_ray);
                path_states.write_beta(pixel_id, beta);
                path_states.write_pdf_bsdf(pixel_id, new_pdf_bsdf);
                path_states.write_data(pixel_id, samples, depth + 1u, 1u);
            };
        };
        surface_nodes[g].define(surface_kernel);
    }

    surface_nodes << to_surface;

    return builder.build();
}

// ============================================================================
// Render loop
// ============================================================================

void WorkGraphPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

    auto &&device = pipeline().device();
    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("No lights in scene. Rendering aborted.");
        return;
    }

    auto spectrum = pipeline().spectrum();
    auto dim = spectrum->node()->dimension();
    LUISA_ASSERT(dim <= 4u, "Spectrum dimension {} exceeds 4.", dim);

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto pixel_count = resolution.x * resolution.y;
    auto max_depth = node<WorkGraphPathTracing>()->max_depth();

    auto tag_to_group = _compute_tag_to_group();
    auto num_groups = _num_groups(tag_to_group);

    LUISA_INFO(
        "Work graph path tracing: resolution={}x{}, spp={}, max_depth={}, "
        "num_groups={}, binning={}",
        resolution.x, resolution.y, spp, max_depth,
        num_groups, node<WorkGraphPathTracing>()->binning_mode()
    );

    // Intermediate buffers
    PathStateSOA path_states { spectrum, pixel_count };
    LightSampleSOA light_samples { spectrum, pixel_count };
    Buffer<Ray> rays = device.create_buffer<Ray>(pixel_count);
    Buffer<Hit> hits = device.create_buffer<Hit>(pixel_count);

    auto active_count_buf = device.create_buffer<uint>(1u);

    auto max_dispatch_groups = (pixel_count + WG_BLOCK_SIZE - 1u) / WG_BLOCK_SIZE;

    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    LUISA_INFO("Building work graph...");
    auto wg = _build_multi_dispatch_graph(
        camera, tag_to_group, num_groups, resolution,
        path_states, light_samples, rays, hits, active_count_buf,
        max_dispatch_groups);

    LUISA_INFO("Compiling work graph...");
    Clock compile_clock;
    auto program = device.compile(wg);
    LUISA_INFO("Work graph compiled in {} ms.", compile_clock.toc());

    // Seed-fill kernel: zero out path_states data
    Kernel1D seed_fill_kernel = [&]() noexcept {
        set_block_size(WG_BLOCK_SIZE, 1u, 1u);
        auto i = dispatch_x();
        $if(i < pixel_count) {
            path_states.write_data(i, 0u, 0u, 0u);
        };
    };
    auto seed_fill = device.compile(seed_fill_kernel);

    command_buffer << synchronize();

    // Render loop
    auto shutter_samples = camera->node()->shutter_samples();
    LUISA_ASSERT(shutter_samples.size() == 1, "no motion blur for now");
    auto s = shutter_samples[0];

    LUISA_INFO("Rendering started.");
    Clock render_clock;
    ProgressBar progress;
    progress.update(0.);

    uint previous_host_active_count = pixel_count;
    uint host_active_count = pixel_count;

    pipeline().update(command_buffer, s.point.time);

    // Zero out path_states and prime the counters
    command_buffer << seed_fill().dispatch(pixel_count)
                   << active_count_buf.copy_from(&host_active_count)
                   << synchronize();

    // Single unified loop: each step is one full wavefront across all in-flight paths.
    // Terminated paths write seeds for their next sample, keeping occupancy high until
    // all spp samples for all pixels are done.
    auto max_iters = spp * max_depth;
    for (auto i = 0u; i < max_iters; i += 1) {
        auto dispatch_groups = max_dispatch_groups;
        WGEntryRecord entry_rec{};
        entry_rec.size = uint3(dispatch_groups, 1u, 1u);
        command_buffer << program().dispatch(1, sizeof(WGEntryRecord), &entry_rec);
        command_buffer << synchronize();
        // command_buffer << active_count_buf.copy_to(&host_active_count)
        //               << synchronize();
        
        // LUISA_INFO("active count: {}", host_active_count);
        // break;
    }

    command_buffer << synchronize();
    progress.done();

    auto render_time = render_clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WorkGraphPathTracing)
