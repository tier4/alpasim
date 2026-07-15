# AlpaSim Physics

This project contains the code for the Physics micro-service of the AlpaSim project, which uses a
mesh of the environment to constrain the motion of simulated agents to the ground surface. It does
not handle collisions or vehicle dynamics.

## Environment Setup

`uv` is used to manage the development environment.

## Running the Sim Service

This package now provides the CARLA-free library modules (`backend`, `ply_io`, `utils`) that the
physics server consumes. The server entry point itself has moved to
`docker/carla/carla_physics_server` and is built into the `docker/carla/physics.Dockerfile` image
(select it in wizard with `physics=carla`).
