use bevy::{
    camera::{Viewport, visibility::RenderLayers}, color::palettes::tailwind, feathers::{
        FeathersPlugins,
        controls::{ButtonProps, SliderProps, button, slider},
        dark_theme::create_dark_theme,
        theme::{ThemedText, UiTheme},
    }, input_focus::InputFocus, prelude::*, ui_widgets::{Activate, SliderPrecision, SliderStep, SliderValue, observe, slider_self_update}, window::{WindowResized, WindowResolution}
};
use bevy_pancam::{PanCam, PanCamPlugin};
use bevy_prototype_lyon::prelude::*;
use bevy_prototype_lyon::{
    plugin::ShapePlugin,
    shapes::{Circle, Line},
};
use itertools::Itertools;
use nalgebra as na;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::ants::AgentMaterial;

mod ants;

#[derive(Component)]
struct SeedSlider;

#[derive(Component)]
struct SizeSlider;

#[derive(Component)]
struct ScaleSlider;

#[derive(Component)]
struct ProbabilitySlider;

#[derive(Message)]
struct RestartMessage;

#[derive(Component, Clone)]
struct Graph {
    nodes: Vec<Vec2>,
    edges: Vec<Vec<usize>>,
}

#[derive(Component)]
struct GraphPhysics {
    velocities: Vec<Vec2>,
}

impl GraphPhysics {
    fn from_graph(graph: &Graph) -> Self {
        Self {
            velocities: vec![Vec2::new(0.0, 0.0); graph.nodes.len()],
        }
    }
}

impl Graph {
    const RADIUS: f32 = 1.0;
    const EDGE_WIDTH: f32 = 0.2;

    fn new_random_seeded(seed: u64, size: usize, p: f32) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Generate directed graph with decreasing arrows (u -> v iif u > v)
        let mut edges = (0..size)
            .map(|u| {
                (0..size)
                    .filter(|&v| u > v && rng.random::<f32>() < p)
                    .collect_vec()
            })
            .collect_vec();

        // Make it undirected by doubling up every arrow (Rust makes this a pain)
        for u in 0..size {
            let (left, right) = edges.split_at_mut(u);
            for &v in &right[0] {
                left[v].push(u);
            }
        }

        Self {
            nodes: (0..size)
                .map(|_| rng.random::<Vec2>() * 100.0)
                .collect_vec(),
            edges,
        }
    }

    fn update_meshes(&self, entity: Entity, mut commands: Commands) {
        let mut edges = ShapeBuilder::new();
        for (u, neighbors) in self.edges.iter().enumerate() {
            for &v in neighbors.iter().filter(|&&v| u < v) {
                edges = edges.add(&Line(self.nodes[u], self.nodes[v]));
            }
        }
        let edges = edges
            .stroke(Stroke::new(Color::WHITE, Self::EDGE_WIDTH))
            .build();

        let mut nodes = ShapeBuilder::new();
        for &p in &self.nodes {
            nodes = nodes.add(&Circle {
                center: p,
                radius: Self::RADIUS,
            });
        }
        let nodes = nodes
            .fill(tailwind::BLUE_500)
            .stroke(Stroke::new(Color::WHITE, Self::EDGE_WIDTH))
            .build();

        commands.entity(entity).despawn_children();
        commands.entity(entity).with_children(|parent| {
            parent.spawn((edges, RenderLayers::layer(1)));
            parent.spawn((
                nodes,
                RenderLayers::layer(1),
                GlobalTransform::from_xyz(0.0, 0.0, 10.0),
            ));
        });
    }

    fn spectral_layout(&mut self, scale: f32) {
        let size = self.nodes.len();
        let laplace_matrix = na::DMatrix::from_fn(size, size, |i, j| {
            if i == j {
                self.edges[i].len() as f32
            } else if self.edges[i].contains(&j) {
                -1.0
            } else {
                0.0
            }
        });
        let eigen = nalgebra_lapack::SymmetricEigen::new(laplace_matrix.clone_owned());
        let (_, x_vec, y_vec) = eigen
            .eigenvectors
            .column_iter()
            .map(|c| {
                let p = laplace_matrix.clone() * c;
                let value = p.component_div(&c).mean();
                (value, c)
            })
            .k_smallest_by(3, |(a, _), (b, _)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, c)| c)
            .collect_tuple()
            .unwrap();

        for i in 0..size {
            self.nodes[i] = Vec2::new(x_vec[i] as f32 * scale, y_vec[i] as f32 * scale);
        }
    }
}

fn update_graph(
    mut commands: Commands,
    query: Query<Entity, With<Graph>>,
    seed: Single<&SliderValue, With<SeedSlider>>,
    size: Single<&SliderValue, With<SizeSlider>>,
    prob: Single<&SliderValue, With<ProbabilitySlider>>,
    scale: Single<&SliderValue, With<ScaleSlider>>,
    mut restart_event_reader: MessageReader<RestartMessage>,
) {
    for _ in restart_event_reader.read() {
        for entity in query {
            commands.entity(entity).despawn();
        }

        let mut graph = Graph::new_random_seeded(seed.0 as u64, size.0 as usize, prob.0);

        //graph.spectral_layout(scale.0);

        let phys = GraphPhysics::from_graph(&graph);

        let aco = ants::ACO::from_graph(graph.clone(), seed.0 as u64);

        commands.spawn((
            graph,
            phys,
            Visibility::default(),
            GlobalTransform::default(),
            children![(aco, Transform::default(), Visibility::default())],
        ));
    }
}

fn update_graph_meshes(mut commands: Commands, query: Query<(Entity, &Graph), Changed<Graph>>) {
    for (entity, graph) in query {
        graph.update_meshes(entity, commands.reborrow());
    }
}

fn toolbar() -> impl Bundle {
    (
        Node {
            width: percent(30.),
            padding: UiRect::all(px(5)),
            row_gap: px(5),
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Center,
            flex_direction: FlexDirection::Column,
            display: Display::Flex,
            ..default()
        },
        BackgroundColor(Color::BLACK),
        BorderColor::all(Color::WHITE),
        children![
            (
                button(
                    ButtonProps::default(),
                    (),
                    Spawn((Text::new("button"), ThemedText))
                ),
                observe(|_activate: On<Activate>, mut restart_message_writer: MessageWriter<RestartMessage>| {
                    restart_message_writer.write(RestartMessage);
                })
            ),
            (
                slider(
                    SliderProps {
                        value: 0.,
                        min: 0.,
                        max: 20.,
                    },
                    (SliderStep(1.), SliderPrecision(0)),
                ),
                observe(slider_self_update),
                SeedSlider,
            ),
            (
                slider(
                    SliderProps {
                        value: 10.,
                        min: 0.,
                        max: 100.,
                    },
                    (SliderStep(1.), SliderPrecision(0)),
                ),
                observe(slider_self_update),
                SizeSlider,
            ),
            (
                slider(
                    SliderProps {
                        value: 0.5,
                        min: 0.,
                        max: 1.,
                    },
                    (SliderStep(0.01), SliderPrecision(2)),
                ),
                observe(slider_self_update),
                ProbabilitySlider,
            ),
            (
                slider(
                    SliderProps {
                        value: 100.0,
                        min: 0.,
                        max: 1000.,
                    },
                    (SliderStep(1.0), SliderPrecision(0)),
                ),
                observe(slider_self_update),
                ScaleSlider,
            ),
        ],
    )
}

fn on_resize(
    // mut ui_cam: Single<&mut Camera, With<IsDefaultUiCamera>>,
    mut graph_cams: Query<&mut Camera, With<GraphCamera>>,
    mut resize_reader: MessageReader<WindowResized>,
    window: Single<&Window>,
) {
    for _ in resize_reader.read() {
        for mut graph_cam in graph_cams.iter_mut() {
            graph_cam.viewport = Some(Viewport {
                physical_size: Vec2::new(
                    window.resolution.physical_width() as f32 * 0.7,
                    window.resolution.physical_height() as f32,
                )
                .as_uvec2(),
                physical_position: Vec2::new(window.resolution.physical_width() as f32 * 0.3, 0.0)
                    .as_uvec2(),
                ..default()
            });
        }
    }
}

#[derive(Component)]
struct GraphCamera;

fn setup(mut commands: Commands, window: Single<&Window>, mut gizmo_assets: ResMut<Assets<GizmoAsset>>) {
    let mut gizmo = GizmoAsset::new();
    let window_size = window.resolution.physical_size().as_vec2();

    gizmo.grid_3d(Isometry3d::from_translation(Vec3::new(50.0, 50.0, 0.0)), UVec3::splat(99), Vec3::splat(10.0), Color::WHITE);

    commands.spawn((Camera2d, IsDefaultUiCamera));
    commands.spawn((
        GraphCamera,
        Camera3d::default(),
        Camera {
            order: 1,
            viewport: Some(Viewport {
                physical_size: Vec2::new(window_size.x * 0.7, window_size.y).as_uvec2(),
                physical_position: Vec2::new(window_size.x * 0.3, 0.0).as_uvec2(),
                ..default()
            }),
            clear_color: ClearColorConfig::Default,
            ..default()
        },
        Transform::from_xyz(50.0, 50., 100.0),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical {
                viewport_height: 100.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        RenderLayers::layer(1),
        Msaa::Sample4,
        PanCam::default(),
    ));
    commands.spawn((
        GraphCamera,
        Camera2d,
        Camera {
            order: 2,
            viewport: Some(Viewport {
                physical_size: Vec2::new(window_size.x * 0.7, window_size.y).as_uvec2(),
                physical_position: Vec2::new(window_size.x * 0.3, 0.0).as_uvec2(),
                ..default()
            }),
            ..default()
        },
        Transform::from_xyz(50.0, 50., 100.0),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical {
                viewport_height: 100.0,
            },
            ..OrthographicProjection::default_2d()
        }),
        RenderLayers::layer(1),
        Msaa::Sample4,
        PanCam::default(),
    ));
    commands.spawn(toolbar());
    commands.spawn((
        Gizmo {
            handle: gizmo_assets.add(gizmo),
            ..default()
        },
        RenderLayers::layer(1),
    ));
}

fn force_based_refine(query: Query<(&mut Graph, &mut GraphPhysics)>) {
    const SPRING_LENGTH: f32 = 500.0;
    const SPRING_STRENGTH: f32 = 0.01;
    const MAGNET_STRENGTH: f32 = 0.04;
    const TIME_STEP: f32 = 0.1;
    const FRICTION: f32 = 0.97;

    for (mut graph, mut phys) in query {
        let size = graph.nodes.len();
        for (u, vel) in phys.velocities.iter_mut().enumerate() {
            let mut forces = Vec2::ZERO;

            for v in (0..size).filter(|&v| v != u) {
                let delta = graph.nodes[u] - graph.nodes[v];
                forces += delta.normalize()
                    * (1.0
                        / (MAGNET_STRENGTH
                            * (delta.length_squared() - 2.0 * Graph::RADIUS * Graph::RADIUS))
                        - 1.0)
                        .clamp(0.0, 100.0)
            }

            for &v in &graph.edges[u] {
                let delta = graph.nodes[v] - graph.nodes[u];
                forces -=
                    delta.normalize() * SPRING_STRENGTH * (SPRING_LENGTH - delta.length_squared());
            }

            *vel += forces * TIME_STEP;
        }

        for (pos, vel) in graph.nodes.iter_mut().zip(phys.velocities.iter_mut()) {
            *pos += *vel * TIME_STEP;
            *vel *= FRICTION;
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::default().with_scale_factor_override(2.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((
            FeathersPlugins,
            ShapePlugin,
            PanCamPlugin,
            MaterialPlugin::<AgentMaterial>::default(),
        ))
        .add_message::<RestartMessage>()
        .insert_resource(UiTheme(create_dark_theme()))
        .init_resource::<InputFocus>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_graph_meshes,
                on_resize,
                update_graph,
                ants::spawn_agents,
                ants::update_agents,
            ),
        )
        // .add_systems(FixedUpdate, force_based_refine)
        .add_systems(FixedUpdate, ants::update_acos)
        .run();
}
