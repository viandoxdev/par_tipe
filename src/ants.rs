use core::f32;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::image::ImageSampler;
use bevy::math::{IVec2, Isometry2d, Vec2, bounding::Aabb2d};
use bevy::mesh::MeshTag;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d};
use itertools::Itertools;
use rand::{Rng, RngCore, SeedableRng};

use crate::Graph;

const SHADER_ASSET_PATH: &str = "shaders/agents.wgsl";

#[derive(Component)]
pub struct ACO {
    config: ACOConfig,
    grid_mapping: Aabb2d,
    colonies: usize,
    graph: Graph,
    edges: Vec<(usize, usize)>,
    state_buffer: AgentsState,
    state_backbuffer: AgentsState,
    rng: Box<dyn RngCore + Sync + Send>,
}

struct ACOConfig {
    agents_per_colony: usize,

    grid_width: usize,
    grid_height: usize,

    pheromone_evaporation_rate: f32,
    pheromone_strength: f32,

    random_steer_strength: f32,

    speed: f32,

    goal_sense_factor: f32,
    goal_radius: f32,

    outsider_avoidance: f32,

    sensor_radius: i32,
    sensor_angle: f32,
    sensor_distance: f32,
}

#[derive(Clone)]
struct AgentsState {
    positions: Vec<Vec2>,
    angles: Vec<f32>,
    walking_home: Vec<bool>,
    pheromones_map: Vec<Box<[f32]>>,
}

impl Default for ACOConfig {
    fn default() -> Self {
        Self {
            agents_per_colony: 50,
            grid_width: 1024,
            grid_height: 1024,
            pheromone_evaporation_rate: 1.0,
            pheromone_strength: 1.0,
            goal_sense_factor: 2.0,
            goal_radius: 2.0,
            random_steer_strength: 1.0,
            speed: 1.0,
            outsider_avoidance: 1.0,
            sensor_radius: 1,
            sensor_angle: f32::consts::PI / 3.0,
            sensor_distance: 0.5,
        }
    }
}

impl AgentsState {
    fn new(
        graph: &Graph,
        edges: &[(usize, usize)],
        agents_per_colony: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            positions: edges
                .iter()
                .flat_map(|&(u, _)| std::iter::repeat_n(graph.nodes[u], agents_per_colony))
                .collect_vec(),
            angles: edges
                .iter()
                .flat_map(|&(u, v)| {
                    std::iter::repeat_n(graph.nodes[u].angle_to(graph.nodes[v]), agents_per_colony)
                })
                .collect_vec(),
            walking_home: vec![false; edges.len() * agents_per_colony],
            pheromones_map: std::iter::repeat_n(
                vec![0.0; edges.len() * 2].into_boxed_slice(),
                width * height,
            )
            .collect_vec(),
        }
    }
}

impl ACO {
    pub fn from_graph(graph: Graph, seed: u64) -> Self {
        let config = ACOConfig::default();
        let colonies = graph.nodes.len();
        let grid_mapping = Aabb2d::from_point_cloud(Isometry2d::IDENTITY, &graph.nodes);
        let edges = graph
            .edges
            .iter()
            .enumerate()
            .flat_map(|(u, n)| n.iter().filter(move |&&v| u < v).map(move |&v| (u, v)))
            .collect_vec();
        let state = AgentsState::new(
            &graph,
            &edges,
            config.agents_per_colony,
            config.grid_width,
            config.grid_height,
        );
        let rng = Box::new(rand_wyrand::WyRand::seed_from_u64(seed));

        Self {
            config,
            grid_mapping,
            colonies,
            graph,
            edges,
            state_buffer: state.clone(),
            state_backbuffer: state,
            rng,
        }
    }

    fn float_to_grid(&self, pos: Vec2) -> IVec2 {
        Vec2::new(
            (pos.x - self.grid_mapping.min.x) / (self.grid_mapping.max.x - self.grid_mapping.min.x)
                * (self.config.grid_width as f32),
            (pos.y - self.grid_mapping.min.y) / (self.grid_mapping.max.y - self.grid_mapping.min.y)
                * (self.config.grid_height as f32),
        )
        .as_ivec2()
    }

    fn flip_state_buffers(&mut self) {
        std::mem::swap(&mut self.state_backbuffer, &mut self.state_buffer);
    }

    fn sense(&self, position: Vec2, angle: f32, colony: usize, back: bool, goal: Vec2) -> f32 {
        let sensor_pos = position + Vec2::from_angle(angle) * self.config.sensor_distance;
        let sensor_grid = self.float_to_grid(sensor_pos);

        let mut weight = 0.0;
        for dx in (-self.config.sensor_radius)..=self.config.sensor_radius {
            for dy in (-self.config.sensor_radius)..=self.config.sensor_radius {
                let pos = sensor_grid + IVec2::new(dx, dy);
                if pos.x >= 0
                    && pos.x < self.config.grid_width as i32
                    && pos.y >= 0
                    && pos.y < self.config.grid_height as i32
                {
                    let pheromones = &self.state_buffer.pheromones_map
                        [pos.x as usize + pos.y as usize * self.config.grid_width];

                    weight += pheromones
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let (i, b) = (i % self.edges.len(), i >= self.edges.len());
                            if (i, b) != (colony, back) {
                                -v.powf(self.config.outsider_avoidance)
                            } else if i != colony {
                                v
                            } else {
                                0.0
                            }
                        })
                        .sum::<f32>();

                    weight += 1.0
                        / (sensor_pos - goal)
                            .length_squared()
                            .powf(self.config.goal_sense_factor * 0.5);
                }
            }
        }

        weight
    }

    fn evaporate_pheromones(&mut self) {
        for (i, p) in self.state_buffer.pheromones_map.iter().enumerate() {
            for (j, &v) in p.iter().enumerate() {
                self.state_backbuffer.pheromones_map[i][j] =
                    (v - self.config.pheromone_evaporation_rate).max(0.0);
            }
        }
    }
    pub fn step(&mut self) {
        // Loop over each colony
        for (colony, &(u, v)) in self.edges.iter().enumerate() {
            // Loop over each agents
            for agent in 0..self.config.agents_per_colony {
                let agent_index = colony * self.config.agents_per_colony + agent;
                let position = self.state_buffer.positions[agent_index];
                let angle = self.state_buffer.angles[agent_index];
                let back = self.state_buffer.walking_home[agent_index];
                let goal = self.graph.nodes[if back { u } else { v }];

                let weight_left = self.sense(
                    position,
                    angle - self.config.sensor_angle,
                    colony,
                    back,
                    goal,
                );
                let weight_front = self.sense(position, angle, colony, back, goal);
                let weight_right = self.sense(
                    position,
                    angle + self.config.sensor_angle,
                    colony,
                    back,
                    goal,
                );

                let random_steer: f32 = self.rng.random();

                let delta_angle = if weight_front > weight_left && weight_front > weight_right {
                    0.0
                } else if weight_front < weight_left && weight_front < weight_right {
                    (random_steer - 0.5) * 2.0 * self.config.random_steer_strength
                } else if weight_right > weight_left {
                    -random_steer * self.config.random_steer_strength
                } else if weight_left < weight_right {
                    random_steer * self.config.random_steer_strength
                } else {
                    // huh
                    0.0
                };

                let new_position = (position + Vec2::from_angle(angle) * self.config.speed)
                    .clamp(self.grid_mapping.min, self.grid_mapping.max);

                {
                    let pos = self.float_to_grid(position);

                    if pos.x >= 0
                        && pos.x < self.config.grid_width as i32
                        && pos.y >= 0
                        && pos.y < self.config.grid_height as i32
                    {
                        let pheromones = &mut self.state_backbuffer.pheromones_map
                            [pos.x as usize + pos.y as usize * self.config.grid_width];

                        pheromones[colony + if back { self.edges.len() } else { 0 }] +=
                            self.config.pheromone_strength;
                    }
                }

                if new_position.distance_squared(goal)
                    < self.config.goal_radius * self.config.goal_radius
                {
                    self.state_backbuffer.walking_home[agent_index] = !back;
                    self.state_backbuffer.angles[agent_index] = goal.angle_to(self.graph.nodes[u]);
                    self.state_backbuffer.positions[agent_index] =
                        (new_position - goal).normalize() * (self.config.goal_radius + 1.0);
                } else {
                    self.state_backbuffer.walking_home[agent_index] = back;
                    self.state_backbuffer.angles[agent_index] = angle + delta_angle;
                    self.state_backbuffer.positions[agent_index] = new_position;
                }
            }
        }

        self.evaporate_pheromones();
        self.flip_state_buffers();
    }

    fn state_to_image(&self, image: &mut Image) {
        let count = self.edges.len();
        let Some(data) = image.data.as_mut() else {
            warn!("Can't write state to an image without backing data");
            return;
        };

        for x in 0..self.config.grid_width {
            for y in 0..self.config.grid_height {
                let index = x + y * self.config.grid_width;
                let pheromones = &self.state_buffer.pheromones_map[index];
                let hue = (0..count)
                    .map(|i| {
                        Vec2::from_angle(i as f32 / count as f32 * f32::consts::PI * 2.0)
                            * (pheromones[i] + pheromones[i + count])
                    })
                    .sum::<Vec2>()
                    .to_angle();
                let color =
                    bevy::color::Hsva::hsv(hue / f32::consts::PI * 180.0, 1.0, 1.0).to_f32_array();
                data[index * 4] = (color[0] * 255.0).clamp(0.0, 255.0) as u8;
                data[index * 4 + 1] = (color[1] * 255.0).clamp(0.0, 255.0) as u8;
                data[index * 4 + 2] = (color[2] * 255.0).clamp(0.0, 255.0) as u8;
                data[index * 4 + 3] = (color[3] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[derive(Component)]
pub struct Agent;

#[derive(Component)]
pub struct PheromoneGrid {
    image_handle: Handle<Image>,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct AgentMaterial {
    #[uniform(0)]
    colony_count: u32,
}

impl Material for AgentMaterial {
    fn vertex_shader() -> bevy::shader::ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn fragment_shader() -> bevy::shader::ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}

pub fn spawn_agents(
    query: Query<(Entity, &ACO), Added<ACO>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AgentMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mesh_handle = meshes.add(Triangle3d::new(
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ));

    for (entity, aco) in query.iter() {
        let mesh_handle = mesh_handle.clone();
        let material_handle = materials.add(AgentMaterial {
            colony_count: aco.edges.len() as u32,
        });

        let size = Extent3d {
            width: aco.config.grid_width as u32,
            height: aco.config.grid_height as u32,
            depth_or_array_layers: 1,
        };

        let mut image = Image::new_fill(
            size,
            bevy::render::render_resource::TextureDimension::D2,
            &[0, 0, 0, 255],
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );

        image.sampler = ImageSampler::nearest();

        let handle = images.add(image);

        commands.entity(entity).with_children(move |spawner| {
            for colony in 0..aco.edges.len() {
                for agent in 0..aco.config.agents_per_colony {
                    let index = colony * aco.config.agents_per_colony + agent;
                    let pos = aco.state_buffer.positions[index];

                    spawner.spawn((
                        Mesh3d(mesh_handle.clone()),
                        MeshMaterial3d(material_handle.clone()),
                        MeshTag(index as u32),
                        Transform::from_rotation(Quat::from_rotation_z(
                            aco.state_buffer.angles[index],
                        ))
                        .with_translation(pos.extend(0.0)),
                        RenderLayers::layer(1),
                    ));
                }
            }

            spawner.spawn((
                Sprite::from_image(handle.clone()),
                Transform::from_scale(Vec3::splat(100.0)).with_translation(Vec3::new(0.0, 0.0, 0.0)),
                PheromoneGrid {
                    image_handle: handle,
                },
                RenderLayers::layer(1),
            ));
        });
    }
}

pub fn update_agents(
    acos: Query<&ACO>,
    mut transforms: Query<(&mut Transform, &MeshTag, &ChildOf), With<Agent>>,
    pheromone_grids: Query<(&PheromoneGrid, &ChildOf)>,
    mut images: ResMut<Assets<Image>>,
) {
    for (mut transform, index, parent) in transforms.iter_mut() {
        let Ok(aco) = acos.get(parent.0) else {
            warn!("Agent is missing parent ACO");
            continue;
        };

        let index = index.0 as usize;
        let pos = aco.state_buffer.positions[index];

        transform.translation = pos.extend(0.0);
        transform.rotation = Quat::from_rotation_z(aco.state_buffer.angles[index]);
    }

    for (grid, parent) in pheromone_grids.iter() {
        let Ok(aco) = acos.get(parent.0) else {
            warn!("PheromoneGrid is missing parent ACO");
            continue;
        };

        let Some(image) = images.get_mut(&grid.image_handle) else {
            warn!("PheromoneGrid image doesn't exist");
            continue;
        };

        aco.state_to_image(image);
    }
}

pub fn update_acos(mut acos: Query<&mut ACO>) {
    for mut aco in acos.iter_mut() {
        aco.step();
    }
}
