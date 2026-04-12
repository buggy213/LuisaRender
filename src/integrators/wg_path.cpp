//
// Work Graph Path Tracing Integrator
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

// ============================================================================
// Record Structs
// ============================================================================

struct IntersectRecord {
    uint pixel_id;
    uint sample_id;
    uint depth;
    float wl_sample;
    float4 beta;
    float pdf_bsdf;
    Ray ray;
};

struct PostIntersectRecord {
    uint pixel_id;
    uint sample_id;
    uint depth;
    float wl_sample;
    float4 beta;
    float pdf_bsdf;
    Ray ray;
    Hit hit;
};

struct SurfaceRecord {
    uint pixel_id;
    uint sample_id;
    uint depth;
    float wl_sample;
    float4 beta;
    Ray ray;
    Hit hit;
    float4 light_emission;
    float3 light_wi;
    float light_pdf;
    uint surface_tag;
};

// Entry record for dynamic dispatch grid sizing
struct WGEntryRecord : DispatchGridRecord {};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::IntersectRecord,
             pixel_id, sample_id, depth, wl_sample,
             beta, pdf_bsdf, ray){};

LUISA_STRUCT(luisa::render::PostIntersectRecord,
             pixel_id, sample_id, depth, wl_sample,
             beta, pdf_bsdf, ray, hit){};

LUISA_STRUCT(luisa::render::SurfaceRecord,
             pixel_id, sample_id, depth, wl_sample,
             beta, ray, hit,
             light_emission, light_wi, light_pdf, surface_tag){};

LUISA_STRUCT(luisa::render::WGEntryRecord, size){};

namespace luisa::render {

// ============================================================================
// Helpers: SampledSpectrum <-> float4
// ============================================================================

static SampledSpectrum float4_to_spectrum(Expr<float4> v, uint dim) noexcept {
    SampledSpectrum s{dim};
    if (dim > 0u) s[UInt(0)] = v.x;
    if (dim > 1u) s[UInt(1)] = v.y;
    if (dim > 2u) s[UInt(2)] = v.z;
    if (dim > 3u) s[UInt(3)] = v.w;
    return s;
}

static Float4 spectrum_to_float4(const SampledSpectrum &s, uint dim) noexcept {
    auto x = dim > 0u ? s[UInt(0)] : def(0.f);
    auto y = dim > 1u ? s[UInt(1)] : def(0.f);
    auto z = dim > 2u ? s[UInt(2)] : def(0.f);
    auto w = dim > 3u ? s[UInt(3)] : def(0.f);
    return make_float4(x, y, z, w);
}

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
        const Buffer<IntersectRecord> &read_buffer,
        const Buffer<IntersectRecord> &write_buffer,
        const Buffer<uint> &write_counter,
        const Buffer<uint> &active_count_buf,
        const Buffer<uint> &debug_counter,
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
    const Buffer<IntersectRecord> &read_buffer,
    const Buffer<IntersectRecord> &write_buffer,
    const Buffer<uint> &write_counter,
    const Buffer<uint> &active_count_buf,
    const Buffer<uint> &debug_counter,
    uint max_dispatch_groups) noexcept {

    auto spectrum = pipeline().spectrum();
    auto dim = spectrum->node()->dimension();
    LUISA_ASSERT(dim <= 4u, "Spectrum dimension {} exceeds 4.", dim);

    auto max_depth = node<WorkGraphPathTracing>()->max_depth();
    auto rr_depth = node<WorkGraphPathTracing>()->rr_depth();
    auto rr_threshold = node<WorkGraphPathTracing>()->rr_threshold();

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

    WorkGraphNodeKernel entry_kernel = [&](Var<WGEntryRecord>) {
        auto thread_id = dispatch_x();
        auto count = active_count_buf->read(0u);

        Var<PostIntersectRecord> post;
        Bool active = thread_id < count;
        Bool miss = def(false);
        Bool has_light = def(false);
        Bool has_surface = def(false);
        $if (active) {
            auto rec = read_buffer->read(thread_id);
            auto ray = rec.ray;
            auto hit = pipeline().geometry()->trace_closest(ray);

            post.pixel_id = rec.pixel_id;
            post.sample_id = rec.sample_id;
            post.depth = rec.depth;
            post.wl_sample = rec.wl_sample;
            post.beta = rec.beta;
            post.pdf_bsdf = rec.pdf_bsdf;
            post.ray = rec.ray;
            post.hit = hit;

            miss = hit->miss();
            $if (!miss) {
                auto shape = pipeline().geometry()->instance(hit.inst);
                has_light = shape.has_light();
                has_surface = shape.has_surface();
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
            if (pipeline().environment()) {
                auto wi = input.ray->direction();
                auto u_wl = input.wl_sample;
                auto swl = spectrum->sample(abs(u_wl));
                $if(u_wl < 0.f) { swl.terminate_secondary(); };
                auto beta = float4_to_spectrum(input.beta, dim);
                auto pdf_bsdf = input.pdf_bsdf;
                auto eval = light_sampler()->evaluate_miss(wi, swl, 0.f);
                auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                auto Li = beta * eval.L * mis_weight;
                auto pixel_coord = make_uint2(input.pixel_id % resolution.x,
                                              input.pixel_id / resolution.x);
                camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
            }
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
            if (!pipeline().lights().empty()) {
                auto ray = input.ray;
                auto hit = input.hit;
                auto u_wl = input.wl_sample;
                auto swl = spectrum->sample(abs(u_wl));
                $if(u_wl < 0.f) { swl.terminate_secondary(); };
                auto beta = float4_to_spectrum(input.beta, dim);
                auto pdf_bsdf = input.pdf_bsdf;
                auto it = pipeline().geometry()->interaction(ray, hit);
                auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, 0.f);
                auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                auto Li = beta * eval.L * mis_weight;
                auto pixel_coord = make_uint2(input.pixel_id % resolution.x,
                                              input.pixel_id / resolution.x);
                camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);

                auto shape = pipeline().geometry()->instance(hit.inst);
                light_to_sample->write(input, shape.has_surface());
            }
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

        auto ray = input.ray;
        auto hit = input.hit;
        auto it = pipeline().geometry()->interaction(ray, hit);
        auto u_wl = input.wl_sample;
        auto swl = spectrum->sample(abs(u_wl));
        $if(u_wl < 0.f) { swl.terminate_secondary(); };

        auto light_sample = light_sampler()->sample(
            *it, u_light_selection, u_light_surface, swl, 0.f);

        auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

        Var<SurfaceRecord> surf_rec;
        surf_rec.pixel_id = input.pixel_id;
        surf_rec.sample_id = input.sample_id;
        surf_rec.depth = input.depth;
        surf_rec.wl_sample = input.wl_sample;
        surf_rec.beta = input.beta;
        surf_rec.ray = input.ray;
        surf_rec.hit = input.hit;

        auto emission_spec = ite(occluded, SampledSpectrum{dim, 0.f}, light_sample.eval.L);
        surf_rec.light_emission = spectrum_to_float4(emission_spec, dim);
        surf_rec.light_wi = light_sample.shadow_ray->direction();
        surf_rec.light_pdf = ite(occluded, 0.f, light_sample.eval.pdf);

        auto surface_tag = it->shape().surface_tag();
        surf_rec.surface_tag = surface_tag;

        // Compile-time tag-to-group lookup
        auto group = def(0u);
        $switch(surface_tag) {
            for (uint t = 0u; t < tag_to_group.size(); t++) {
                $case(t) { group = uint(tag_to_group[t]); };
            }
        };

        // Always output — surface node still needs to sample next bounce even if occluded
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
            // DEBUG: unconditional bump — did the surface node execute at all?
            debug_counter->atomic(0u).fetch_add(1u);
            auto pixel_id = input.pixel_id;
            sampler()->load_state(pixel_id);
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            $if(input.depth + 1u >= rr_depth) {
                u_rr = sampler()->generate_1d();
            };
            sampler()->save_state(pixel_id);

            auto ray = input.ray;
            auto hit = input.hit;
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto u_wl = input.wl_sample;
            auto swl = spectrum->sample(abs(u_wl));
            $if(u_wl < 0.f) { swl.terminate_secondary(); };
            auto beta = float4_to_spectrum(input.beta, dim);
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            auto wo = -ray->direction();
            auto out_wl_sample = def(u_wl);

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
                        out_wl_sample = -abs(u_wl);
                    };
                }

                // Direct lighting
                auto light_wi = input.light_wi;
                auto pdf_light = input.light_pdf;
                $if(pdf_light > 0.f) {
                    auto eval = closure->evaluate(wo, light_wi);
                    auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                    auto Ld = float4_to_spectrum(input.light_emission, dim);
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
                $if(input.depth + 1u >= rr_depth) {
                    terminated = q < rr_threshold & u_rr >= q;
                    beta *= ite(q < rr_threshold, 1.f / q, 1.f);
                };
            };
            $if(input.depth + 1u >= max_depth) {
                terminated = true;
            };

            // Write continuation to write_buffer
            $if(!terminated) {
                Var<IntersectRecord> cont;
                cont.pixel_id = pixel_id;
                cont.sample_id = input.sample_id;
                cont.depth = input.depth + 1u;
                cont.wl_sample = out_wl_sample;
                cont.beta = spectrum_to_float4(beta, dim);
                cont.pdf_bsdf = new_pdf_bsdf;
                cont.ray = new_ray;

                auto slot = write_counter->atomic(0u).fetch_add(1u);
                write_buffer->write(slot, cont);
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

    // Buffers: entry always reads from read_buf, surface always writes to write_buf
    auto read_buf = device.create_buffer<IntersectRecord>(pixel_count);
    auto write_buf = device.create_buffer<IntersectRecord>(pixel_count);
    auto write_counter = device.create_buffer<uint>(1u);
    auto active_count_buf = device.create_buffer<uint>(1u);
    auto debug_counter = device.create_buffer<uint>(1u);

    auto max_dispatch_groups = (pixel_count + WG_BLOCK_SIZE - 1u) / WG_BLOCK_SIZE;

    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    LUISA_INFO("Building work graph...");
    auto wg = _build_multi_dispatch_graph(
        camera, tag_to_group, num_groups, resolution,
        read_buf, write_buf, write_counter, active_count_buf,
        debug_counter, max_dispatch_groups);

    LUISA_INFO("Compiling work graph...");
    Clock compile_clock;
    auto program = device.compile(wg);
    LUISA_INFO("Work graph compiled in {} ms.", compile_clock.toc());

    // Ray generation kernel: fills read_buf with initial camera rays
    Kernel1D ray_gen_kernel = [&](UInt base_spp, Float time, Float shutter_weight) noexcept {
        set_block_size(WG_BLOCK_SIZE, 1u, 1u);
        auto pixel_id = dispatch_x();
        $if (pixel_id < pixel_count) {
            auto pixel_coord = make_uint2(pixel_id % resolution.x,
                                          pixel_id / resolution.x);
            sampler()->start(pixel_coord, base_spp);
            auto u_filter = sampler()->generate_pixel_2d();
            auto u_lens = camera->node()->requires_lens_sampling()
                              ? sampler()->generate_2d()
                              : make_float2(.5f);
            auto u_wavelength = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
            sampler()->save_state(pixel_id);

            auto camera_sample = camera->generate_ray(pixel_coord, time, u_filter, u_lens);

            Var<IntersectRecord> rec;
            rec.pixel_id = pixel_id;
            rec.sample_id = base_spp;
            rec.depth = 0u;
            rec.wl_sample = u_wavelength;
            rec.beta = spectrum_to_float4(
                SampledSpectrum{dim, shutter_weight * camera_sample.weight}, dim);
            rec.pdf_bsdf = 1e16f;
            rec.ray = camera_sample.ray;

            read_buf->write(pixel_id, rec);
            // camera->film()->accumulate(pixel_coord, make_float3(0.f), 1.f);
        };
    };
    auto ray_gen = device.compile(ray_gen_kernel);

    command_buffer << synchronize();

    // Render loop
    auto shutter_samples = camera->node()->shutter_samples();
    LUISA_INFO("Rendering started.");
    Clock render_clock;
    ProgressBar progress;
    progress.update(0.);

    uint host_active_count = 0u;
    uint zero = 0u;

    auto sample_id = 0u;

    LUISA_ASSERT(shutter_samples.size() == 1, "no motion blur for now");
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);

        for (auto i = 0u; i < 1/* s.spp */; i++) {
            // Generate initial rays
            command_buffer << write_counter.copy_from(&zero);
            command_buffer << ray_gen(sample_id, s.point.time, s.point.weight)
                                  .dispatch(pixel_count);

            auto active_count = pixel_count;
            host_active_count = active_count;
            command_buffer << active_count_buf.copy_from(&host_active_count);
            command_buffer << synchronize();

            // Bounce loop with readback
            for (auto bounce = 0u; bounce < max_depth; bounce++) {
                auto dispatch_groups = (active_count + WG_BLOCK_SIZE - 1u) / WG_BLOCK_SIZE;
                WGEntryRecord entry_rec{};
                entry_rec.size = uint3(dispatch_groups, 1u, 1u);
                command_buffer << program().dispatch(1, sizeof(WGEntryRecord), &entry_rec);

                // Readback debug_counter to see how many paths actually reached shading node
                uint h_debug_counter = 0;
                command_buffer << debug_counter.copy_to(&h_debug_counter);
                command_buffer << synchronize();

                printf("bounce = %d, h_debug_counter = %d\n", bounce, h_debug_counter);

                // Readback write_counter to see how many paths survive
                command_buffer << write_counter.copy_to(&host_active_count);
                command_buffer << synchronize();

                LUISA_INFO("sample {}, bounce {}: host_active_count = {}", sample_id, bounce, host_active_count);

                if (host_active_count == 0u) break;

                // Copy surviving continuations from write_buf -> read_buf
                command_buffer << read_buf.view(0, host_active_count)
                                      .copy_from(write_buf.view(0, host_active_count));

                // Reset write counter and update active count for next bounce
                command_buffer << write_counter.copy_from(&zero);
                active_count = host_active_count;
                host_active_count = active_count;
                command_buffer << active_count_buf.copy_from(&host_active_count);
            }

            sample_id++;
            auto p = sample_id / static_cast<double>(spp);
            progress.update(p);
        }
    }

    command_buffer << synchronize();
    progress.done();

    auto render_time = render_clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WorkGraphPathTracing)
