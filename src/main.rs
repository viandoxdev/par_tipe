use bevy::{
    camera::{Viewport, visibility::RenderLayers},
    color::palettes::tailwind,
    feathers::{
        FeathersPlugins,
        controls::{ButtonProps, SliderProps, button, slider},
        dark_theme::create_dark_theme,
        theme::{ThemedText, UiTheme},
    },
    input_focus::InputFocus,
    prelude::*,
    ui_widgets::{Activate, SliderPrecision, SliderStep, observe, slider_self_update},
    window::WindowResolution,
};
use bevy_prototype_lyon::prelude::*;
use bevy_prototype_lyon::{
    plugin::ShapePlugin,
    shapes::{Circle, Line},
};
use itertools::Itertools;
use rand::{Rng, SeedableRng, rngs::SmallRng};

#[derive(Component)]
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
    const RADIUS: f32 = 5.0;
    const EDGE_WIDTH: f32 = 1.0;

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
                observe(|_activate: On<Activate>| {
                    info!("clicked");
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
                observe(slider_self_update)
            ),
        ],
    )
}

fn setup(mut commands: Commands, window: Single<&Window>) {
    let window_size = window.resolution.physical_size().as_vec2();

    commands.spawn((Camera2d, IsDefaultUiCamera));
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            viewport: Some(Viewport {
                physical_size: (window_size * 0.7).as_uvec2(),
                physical_position: (window_size * 0.3).as_uvec2(),
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
    ));
    commands.spawn(toolbar());
    let graph = Graph::new_random_seeded(2, 30, 0.1);
    let phys = GraphPhysics::from_graph(&graph);
    commands.spawn((graph, phys, Visibility::default(), GlobalTransform::default()));
}

fn force_based_refine(query: Query<(&mut Graph, &mut GraphPhysics)>) {
    const LENGTH_SQUARED: f32 = 500.0;
    const SPRING_STRENGTH: f32 = 0.1;
    const MAGNET_STRENGTH: f32 = 6000.0;
    const TIME_STEP: f32 = 0.01;
    const FRICTION: f32 = 0.97;

    for (mut graph, mut phys) in query {
        let size = graph.nodes.len();
        for (u, vel) in phys.velocities.iter_mut().enumerate() {
            let mut forces = Vec2::ZERO;

            for v in (0..size).filter(|&v| v != u) {
                let delta = graph.nodes[u] - graph.nodes[v];
                if delta.length_squared() < 2.0 * Graph::RADIUS * Graph::RADIUS {
                    forces += delta.normalize() * MAGNET_STRENGTH;
                }
                // forces += delta.normalize() * (MAGNET_STRENGTH * Graph::RADIUS / delta.length_squared());
            }

            for &v in &graph.edges[u] {
                let delta = graph.nodes[v] - graph.nodes[u];
                forces +=
                    delta.normalize() * SPRING_STRENGTH * (delta.length_squared() - LENGTH_SQUARED)
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
        .add_plugins((FeathersPlugins, ShapePlugin))
        .insert_resource(UiTheme(create_dark_theme()))
        .init_resource::<InputFocus>()
        .add_systems(Startup, setup)
        .add_systems(Update, update_graph_meshes)
        .add_systems(FixedUpdate, force_based_refine)
        .run();
}
