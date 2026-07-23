// Lightmapped world geometry + procedural gradient sky.

struct Globals {
    view_proj: mat4x4f,
    cam_pos: vec4f,        // xyz = camera position, w = time (seconds)
    sky_zenith: vec4f,
    sky_horizon: vec4f,
    sun_dir: vec4f,        // xyz = towards sun
    sun_color: vec4f,
}

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var t_albedo: texture_2d<f32>;
@group(1) @binding(1) var s_albedo: sampler;
@group(1) @binding(2) var t_lightmap: texture_2d<f32>;
@group(1) @binding(3) var s_lightmap: sampler;

struct VsIn {
    @location(0) pos: vec3f,
    @location(1) uv: vec2f,
    @location(2) lm_uv: vec2f,
}

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) lm_uv: vec2f,
    @location(2) world_pos: vec3f,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4f(in.pos, 1.0);
    out.uv = in.uv;
    out.lm_uv = in.lm_uv;
    out.world_pos = in.pos;
    return out;
}

fn shade(in: VsOut, albedo: vec4f) -> vec4f {
    let lm = textureSample(t_lightmap, s_lightmap, in.lm_uv).rgb;
    // GoldSrc-style overbright: lightmaps store 0..1 with 1.0 ~= 2x white.
    var color = albedo.rgb * lm * 2.0;
    // Gentle contrast and saturation restore definition to legacy 8-bit
    // materials without bleaching their sunlit midtones.
    color = max((color - vec3f(0.5)) * 1.07 + vec3f(0.5), vec3f(0.0));
    color = mix(vec3f(dot(color, vec3f(0.2126, 0.7152, 0.0722))), color, 1.08);

    // Atmospheric perspective keeps the large arena from reading as an
    // unbounded flat test map and ties distant surfaces into the sky.
    let distance_to_camera = length(in.world_pos - globals.cam_pos.xyz);
    let fog_amount = smoothstep(1550.0, 3900.0, distance_to_camera) * 0.46;
    color = mix(color, globals.sky_horizon.rgb, fog_amount);
    return vec4f(color, 1.0);
}

@fragment
fn fs_opaque(in: VsOut) -> @location(0) vec4f {
    let albedo = textureSample(t_albedo, s_albedo, in.uv);
    return shade(in, albedo);
}

@fragment
fn fs_alphatest(in: VsOut) -> @location(0) vec4f {
    let albedo = textureSample(t_albedo, s_albedo, in.uv);
    if albedo.a < 0.5 {
        discard;
    }
    return shade(in, albedo);
}

// Sky brush faces: ignore surface detail, shade by view ray direction.
fn sky_hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn sky_noise(p: vec2f) -> f32 {
    let cell = floor(p);
    let local = fract(p);
    let blend = local * local * (3.0 - 2.0 * local);
    return mix(
        mix(sky_hash(cell), sky_hash(cell + vec2f(1.0, 0.0)), blend.x),
        mix(sky_hash(cell + vec2f(0.0, 1.0)), sky_hash(cell + vec2f(1.0, 1.0)), blend.x),
        blend.y,
    );
}

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4f {
    let ray = normalize(in.world_pos - globals.cam_pos.xyz);
    let up = clamp(ray.y, 0.0, 1.0);
    let horizon_blend = pow(1.0 - up, 3.0);
    var color = mix(globals.sky_zenith.rgb, globals.sky_horizon.rgb, horizon_blend);

    // Two broad, slowly drifting noise octaves suggest thin desert cloud
    // cover without requiring a cubemap or extra texture memory.
    let cloud_uv = ray.xz / max(ray.y + 0.22, 0.22) * 1.35
        + vec2f(globals.cam_pos.w * 0.004, globals.cam_pos.w * 0.0015);
    let cloud_field = sky_noise(cloud_uv) * 0.68 + sky_noise(cloud_uv * 2.07 + 19.4) * 0.32;
    let cloud = smoothstep(0.54, 0.73, cloud_field) * smoothstep(0.02, 0.3, up);
    color = mix(color, vec3f(0.94, 0.91, 0.84), cloud * 0.14);

    // Warm aerial glow close to the horizon.
    color += vec3f(0.18, 0.09, 0.035) * pow(1.0 - up, 8.0) * 0.35;
    // Sun disc + halo.
    let sun_amount = max(dot(ray, normalize(globals.sun_dir.xyz)), 0.0);
    color += globals.sun_color.rgb * (pow(sun_amount, 420.0) * 1.35 + pow(sun_amount, 10.0) * 0.16);
    return vec4f(color, 1.0);
}
