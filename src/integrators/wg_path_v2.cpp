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

#include "backends/ext/native_resource_ext.hpp"
#include "core/basic_traits.h"
#include "core/basic_types.h"
#include "dsl/builtin.h"
#include "dsl/var.h"
#include "dsl/work_graph/work_graph_types.h"
#include "runtime/stream.h"

namespace luisa::render {

using namespace compute;

static constexpr uint WG_BLOCK_SIZE = 64u;

struct IntersectRecord {
    float4 beta;
    Ray ray;
    Hit hit;
    uint subsample_depth;
    uint thread_id;
    float wl_sample;
    float pdf_bsdf;
};

struct EntryRecord : DispatchGridRecord {};

struct WorkGraphIntegratorState {
    explicit WorkGraphIntegratorState(Device &device, size_t size) noexcept 
      : beta { device.create_buffer<float4>(size) },
        ray { device.create_buffer<Ray>(size) },
        subsample_depth { device.create_buffer<uint>(size) },
        wl_sample { device.create_buffer<float>(size) },
        pdf_bsdf { device.create_buffer<float>(size) },
        live_paths { device.create_buffer<uint>(1) } {}
    
    // Path state SoA buffers
    Buffer<float4> beta;
    Buffer<Ray> ray;
    Buffer<uint> subsample_depth;
    Buffer<float> wl_sample;
    Buffer<float> pdf_bsdf;

    // occasionally read back by CPU to determine when to stop
    Buffer<uint> live_paths;
};

static std::pair<UInt, UInt> unpack_subsample_depth(UInt subsample_depth) {
    return { subsample_depth >> 16, subsample_depth & 0xFFFF };
}

static UInt pack_subsample_depth(UInt subsample, UInt depth) {
    return (subsample << 16) | depth;
}

}// namespace luisa::render

LUISA_STRUCT(luisa::render::EntryRecord,
             size) {};

LUISA_STRUCT(luisa::render::IntersectRecord,
             beta, ray, hit,
             subsample_depth, thread_id, wl_sample,
             pdf_bsdf) {};

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

class WorkGraphPathTracing final : public Integrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    bool _unrolled;
    luisa::string _binning_mode;

public:
    WorkGraphPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
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

class WorkGraphPathTracingInstance final : public Integrator::Instance {

public:
    using Integrator::Instance::Instance;

protected:
    void render(Stream &stream) noexcept override;

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
        uint2 resolution,
        const WorkGraphIntegratorState &state) noexcept;

    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera);
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
    uint2 resolution,
    const WorkGraphIntegratorState &state) noexcept {

    auto spectrum = pipeline().spectrum();
    auto dim = spectrum->node()->dimension();
    LUISA_ASSERT(dim <= 4u, "Spectrum dimension {} exceeds 4.", dim);

    auto max_depth = node<WorkGraphPathTracing>()->max_depth();
    auto rr_depth = node<WorkGraphPathTracing>()->rr_depth();
    auto rr_threshold = node<WorkGraphPathTracing>()->rr_threshold();
    auto spp = camera->node()->spp();

    bool has_environment_light = pipeline().environment() != nullptr;
    bool has_localized_light = !pipeline().lights().empty();

    WorkGraphBuilder builder {"wg-path-bounce"};

    auto entry = builder.add_node<WorkGraphLaunchType::BROADCASTING, EntryRecord>("entry");
    entry.set_threadgroup_size({16, 8, 1});
    auto max_dispatch_groups_x = (resolution.x + 16 - 1) / 16;
    auto max_dispatch_groups_y = (resolution.y + 16 - 1) / 16;
    entry.set_max_dispatch_size({max_dispatch_groups_x, max_dispatch_groups_y, 1});

    auto to_material_eval = entry.array_output<IntersectRecord>(1);
    WorkGraphNodeKernel entry_kernel = [&](Var<EntryRecord> entry_rec) {
        auto thread_x = dispatch_x();
        auto thread_y = dispatch_y();
        auto thread_id = thread_y * resolution.x + thread_x;
        auto pixel_coord = UInt2(thread_x, thread_y);
        auto subsample_depth = state.subsample_depth->read(thread_id);
        auto [subsample, depth] = unpack_subsample_depth(subsample_depth);
        auto active = thread_x < resolution.x & thread_y < resolution.y & subsample < spp;
        
        Var<IntersectRecord> record;
        auto eval_material = Bool(true);
        auto material_group = def(0u);

        $if(active) {
            $if(depth == 0u) {
                state.live_paths->atomic(0).fetch_add(1u);

                // regenerate ray
                sampler()->start(pixel_coord, subsample);
                auto camera_sample = camera->generate_ray(
                    pixel_coord, 
                    0.0f, 
                    sampler()->generate_pixel_2d(), 
                    camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(0.0f)
                );

                record.beta = make_float4(camera_sample.weight);
                record.ray = camera_sample.ray;
                record.subsample_depth = pack_subsample_depth(subsample, depth);
                record.thread_id = thread_id;
                record.wl_sample = spectrum->node()->is_fixed() ? 0.5f : sampler()->generate_1d();
                record.pdf_bsdf = 1e16f;
                sampler()->save_state(thread_id);

                auto swl = spectrum->sample(record.wl_sample);
                camera->film()->accumulate(pixel_coord, make_float3(0.0f), 1.0f);
            } $else {
                // load path from SoA
                record.beta = state.beta->read(thread_id);
                record.ray = state.ray->read(thread_id);
                record.subsample_depth = subsample_depth;
                record.thread_id = thread_id;
                record.wl_sample = state.wl_sample->read(thread_id);
                record.pdf_bsdf = state.pdf_bsdf->read(thread_id);
            };

            auto ray = record.ray;
            auto wi = record.ray->direction();
            auto u_wl = record.wl_sample;
            auto swl = spectrum->sample(abs(u_wl));
            auto beta = float4_to_spectrum(record.beta, dim);
            auto pdf_bsdf = record.pdf_bsdf;

            auto hit = pipeline().geometry()->trace_closest(record.ray);
            $if(hit->miss()) {
                if (pipeline().environment()) {
                    $if(u_wl < 0.f) { swl.terminate_secondary(); };
                    auto eval = light_sampler()->evaluate_miss(wi, swl, 0.f);
                    auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                    auto Li = beta * eval.L * mis_weight;
                    camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);
                }

                // path terminates on miss
                eval_material = false;
            } $else {
                auto shape = pipeline().geometry()->instance(hit.inst);
                auto has_light = shape.has_light();
                auto has_surface = shape.has_surface();
                auto it = pipeline().geometry()->interaction(ray, hit);
                
                $if(has_light) {
                    $if(u_wl < 0.f) { swl.terminate_secondary(); };
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, 0.f);
                    auto mis_weight = balance_heuristic(pdf_bsdf, eval.pdf);
                    auto Li = beta * eval.L * mis_weight;
                    camera->film()->accumulate(pixel_coord, spectrum->srgb(swl, Li), 0.f);

                    // path terminates here for pure emitters
                    eval_material = has_surface;
                };

                $if(has_surface) {
                    $switch(shape.surface_tag()) {
                        for (uint t = 0u; t < tag_to_group.size(); t++) {
                            $case(t) { material_group = uint(tag_to_group[t]); };
                        }
                    };
                };
            };

            $if(!eval_material) {
                state.subsample_depth->write(thread_id, pack_subsample_depth(subsample + 1, 0));
            };
        };

        to_material_eval.write(record, material_group, active & eval_material);
    };
    entry.define(entry_kernel);

    auto num_material_groups = _num_groups(tag_to_group);
    auto material_eval_light_sample = builder.add_node_array<WorkGraphLaunchType::THREAD, IntersectRecord>("material_eval_light_sample", num_material_groups);
    for (uint g = 0u; g < num_material_groups; g += 1) {
        WorkGraphNodeKernel material_eval_light_sample_kernel = [&](Var<IntersectRecord> input) {
            auto thread_id = input.thread_id;
            auto [subsample, depth] = unpack_subsample_depth(input.subsample_depth);
            
            // prepare random variables
            sampler()->load_state(thread_id);
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            $if(depth + 1u >= rr_depth) {
                u_rr = sampler()->generate_1d();
            };
            sampler()->save_state(thread_id);

            auto surface_to_eval = pipeline().surfaces().impl(g);

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
            auto new_ray = def(ray);
            auto new_pdf_bsdf = def(0.f);
            auto terminated = def(false);

            // sample lights
            auto light_sample = light_sampler()->sample(
                *it, 
                u_light_selection, 
                u_light_surface, 
                swl, 
                0.0f
            );
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            auto closure = surface_to_eval->create_closure(swl, 0.0f);
            surface_to_eval->populate_closure(closure.get(), *it, wo, 1.0f);
            closure->pre_eval();
            $outline {
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) {
                        swl.terminate_secondary();
                        out_wl_sample = -abs(u_wl);
                    };
                }

                // Direct lighting
                auto light_wi = light_sample.shadow_ray->direction();
                auto pdf_light = light_sample.eval.pdf;
                $if(pdf_light > 0.0f & !occluded) {
                    auto eval = closure->evaluate(wo, light_wi);
                    auto mis_weight = balance_heuristic(pdf_light, eval.pdf);
                    auto Li = mis_weight / pdf_light * beta * eval.f * light_sample.eval.L;
                    auto pixel_coord = make_uint2(thread_id % resolution.x,
                                                    thread_id / resolution.x);
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
            };
            closure->post_eval();


            // Prepare for next bounce
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.0f; })) {
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

            $if(!terminated) {
                state.subsample_depth->write(thread_id, pack_subsample_depth(subsample, depth + 1));
            };

            $if(terminated) {
                state.subsample_depth->write(thread_id, pack_subsample_depth(subsample + 1, 0));
            };
        };

        material_eval_light_sample[g].define(material_eval_light_sample_kernel);
    }
    material_eval_light_sample << to_material_eval;

    return builder.build();
}

// ============================================================================
// Render loop
// ============================================================================

void WorkGraphPathTracingInstance::render(Stream &stream) noexcept {
    CommandBuffer command_buffer {&stream} ;
    for (auto i = 0u; i < pipeline().camera_count(); i++) {
        auto camera = pipeline().camera(i);
        auto resolution = camera->film()->node()->resolution();
        auto pixel_count = resolution.x * resolution.y;
        camera->film()->prepare(command_buffer);
        {
            _render_one_camera(command_buffer, camera);
            luisa::vector<float4> pixels(pixel_count);
            camera->film()->download(command_buffer, pixels.data());
            command_buffer << compute::synchronize();
            auto film_path = camera->node()->file();
            save_image(film_path, reinterpret_cast<const float *>(pixels.data()), resolution);
        }
        camera->film()->release();
    }
}

void WorkGraphPathTracingInstance::_render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) {
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

    // Create integrator state
    WorkGraphIntegratorState state { device, resolution.x * resolution.y };

    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    LUISA_INFO("Building work graph...");
    auto wg = _build_multi_dispatch_graph(camera, tag_to_group, resolution, state);

    LUISA_INFO("Compiling work graph...");
    Clock compile_clock;
    auto program = device.compile(wg);
    LUISA_INFO("Work graph compiled in {} ms.", compile_clock.toc());

    // Render loop
    auto shutter_samples = camera->node()->shutter_samples();
    LUISA_ASSERT(shutter_samples.size() == 1, "no motion blur for now");
    auto s = shutter_samples[0];

    LUISA_INFO("Rendering started.");
    Clock render_clock;
    ProgressBar progress;
    progress.update(0.);

    pipeline().update(command_buffer, s.point.time);

    // make GPU input record
    auto dispatch_groups = (pixel_count + WG_BLOCK_SIZE - 1u) / WG_BLOCK_SIZE;
    EntryRecord h_entry_rec { DispatchGridRecord(uint3(dispatch_groups, 1u, 1u)) };
    Buffer<EntryRecord> d_entry_rec = device.create_buffer<EntryRecord>(1);

    auto native_res = device.extension<NativeResourceExt>();

    struct GPUInput {
        uint entrypoint_index;
        uint num_records;
        uint64_t record_va;
        uint64_t stride;
    } h_gpu_input {
        .entrypoint_index = 0,
        .num_records = 1,
        .record_va = native_res->get_device_address(d_entry_rec),
        .stride = sizeof(EntryRecord),
    };
    Buffer<GPUInput> d_gpu_input = device.create_buffer<GPUInput>(1);
    command_buffer << d_entry_rec.copy_from(&h_entry_rec) << d_gpu_input.copy_from(&h_gpu_input) << synchronize();

    uint64_t d_gpu_input_addr = native_res->get_device_address(d_gpu_input);

    // rendering loop
    uint paths_started = 0;
    uint total_paths = spp * pixel_count;
    while (true) {
        for (uint i = 0; i < max_depth; i += 1) {
            command_buffer << program().dispatch(d_gpu_input_addr);
        }

        // this technically isn't fully accurate since live_paths is really more 
        // "how many paths started in the previous `max_depth` iters"
        // but it should be close enough for host to tell when its ok to stop integrating
        command_buffer << state.live_paths.copy_to(&paths_started)
                       << synchronize();

        if (paths_started == 0u) break;
        
        progress.update(paths_started / (double)total_paths);
    }

    command_buffer << synchronize();
    progress.done();

    auto render_time = render_clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}
    

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::WorkGraphPathTracing)