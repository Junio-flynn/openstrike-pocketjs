//! Skeletal animation: clips of TRS channels sampled onto a node hierarchy.

use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct NodeTrs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl NodeTrs {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
}

pub struct Channel {
    pub node: usize,
    pub path: ChannelPath,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    /// 3 floats per key for T/S, 4 for R (xyzw).
    pub values: Vec<f32>,
}

impl Channel {
    fn key_span(&self, t: f32) -> (usize, usize, f32) {
        let times = &self.times;
        if times.is_empty() {
            return (0, 0, 0.0);
        }
        if t <= times[0] {
            return (0, 0, 0.0);
        }
        let last = times.len() - 1;
        if t >= times[last] {
            return (last, last, 0.0);
        }
        let hi = times.partition_point(|&k| k <= t);
        let lo = hi - 1;
        let span = times[hi] - times[lo];
        let f = if span > 0.0 {
            (t - times[lo]) / span
        } else {
            0.0
        };
        (lo, hi, f)
    }

    fn vec3_at(&self, key: usize) -> Vec3 {
        let i = key * 3;
        Vec3::new(self.values[i], self.values[i + 1], self.values[i + 2])
    }

    fn quat_at(&self, key: usize) -> Quat {
        let i = key * 4;
        Quat::from_xyzw(
            self.values[i],
            self.values[i + 1],
            self.values[i + 2],
            self.values[i + 3],
        )
        .normalize()
    }

    pub fn sample(&self, t: f32, out: &mut NodeTrs) {
        let (lo, hi, f) = self.key_span(t);
        let f = if self.interpolation == Interpolation::Step {
            0.0
        } else {
            f
        };
        match self.path {
            ChannelPath::Translation => {
                out.translation = self.vec3_at(lo).lerp(self.vec3_at(hi), f);
            }
            ChannelPath::Scale => {
                out.scale = self.vec3_at(lo).lerp(self.vec3_at(hi), f);
            }
            ChannelPath::Rotation => {
                out.rotation = self.quat_at(lo).slerp(self.quat_at(hi), f);
            }
        }
    }
}

pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// A node hierarchy with rest-pose TRS, in evaluation order (parents first).
pub struct Skeleton {
    /// Parent index per node (usize::MAX = root).
    pub parents: Vec<usize>,
    pub rest: Vec<NodeTrs>,
    /// Indices into nodes, ordered so parents precede children.
    pub order: Vec<usize>,
}

impl Skeleton {
    fn sampled_locals(&self, clip: Option<&Clip>, t: f32, looping: bool) -> Vec<NodeTrs> {
        let mut locals = self.rest.clone();
        if let Some(clip) = clip {
            let t = if clip.duration > 0.0 {
                if looping {
                    t.rem_euclid(clip.duration)
                } else {
                    t.clamp(0.0, clip.duration)
                }
            } else {
                0.0
            };
            for channel in &clip.channels {
                if channel.node < locals.len() {
                    channel.sample(t, &mut locals[channel.node]);
                }
            }
        }
        locals
    }

    fn globals_from_locals(&self, locals: &[NodeTrs], globals: &mut Vec<Mat4>) {
        globals.clear();
        globals.resize(locals.len(), Mat4::IDENTITY);
        for &i in &self.order {
            let local = locals[i].matrix();
            globals[i] = if self.parents[i] == usize::MAX {
                local
            } else {
                globals[self.parents[i]] * local
            };
        }
    }

    /// Sample `clip` at `t` (wrapping if `looping`) and return global
    /// transforms per node.
    pub fn global_transforms(
        &self,
        clip: Option<&Clip>,
        t: f32,
        looping: bool,
        globals: &mut Vec<Mat4>,
    ) {
        let locals = self.sampled_locals(clip, t, looping);
        self.globals_from_locals(&locals, globals);
    }

    /// Cross-fade two sampled clips in local bone space before rebuilding the
    /// hierarchy. Quaternion slerp avoids the shearing produced by blending
    /// final joint matrices.
    pub fn global_transforms_blended(
        &self,
        from: Option<&Clip>,
        from_t: f32,
        from_looping: bool,
        to: Option<&Clip>,
        to_t: f32,
        to_looping: bool,
        factor: f32,
        globals: &mut Vec<Mat4>,
    ) {
        let from_locals = self.sampled_locals(from, from_t, from_looping);
        let to_locals = self.sampled_locals(to, to_t, to_looping);
        let factor = factor.clamp(0.0, 1.0);
        let locals: Vec<_> = from_locals
            .iter()
            .zip(to_locals.iter())
            .map(|(from, to)| NodeTrs {
                translation: from.translation.lerp(to.translation, factor),
                rotation: from.rotation.slerp(to.rotation, factor),
                scale: from.scale.lerp(to.scale, factor),
            })
            .collect();
        self.globals_from_locals(&locals, globals);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnimState {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            clip: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}

impl AnimState {
    pub fn advance(&mut self, dt: f32) {
        self.time += dt * self.speed;
    }
}
