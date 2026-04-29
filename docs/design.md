# Design Notes

## Purpose

This document records the design intent of the project at a conceptual level. It is not a line-by-line implementation reference. Its role is to explain how the project is meant to evolve from a small manipulation environment into a broader research platform for advanced learning-based robotic arm control.

## Design Philosophy

The project is built around a simple principle:

robotic manipulation should not be treated as a single flat control problem when the long-term goal is adaptive, general-purpose intelligence.

Instead, the system should be designed in a way that can gradually support:

- low-level motion execution
- skill-level control
- high-level decision making
- perception-grounded action
- human-guided learning

The current manipulation setup is therefore not only a task environment. It is the initial substrate for a larger layered architecture.

## Architectural Direction

The intended architecture can be understood in four conceptual layers.

### 1. Simulation and Task Layer

This layer defines the physical world in which the robotic arm operates.

Responsibilities:

- robot embodiment
- object definitions
- scene layout
- contact and dynamics
- task initialization and reset conditions

This layer should remain interpretable and easy to modify, because future work may involve:

- more objects
- clutter
- randomized scenes
- sim-to-real oriented perturbations

### 2. Environment Interface Layer

This layer exposes the manipulation task through a reusable learning interface.

Responsibilities:

- observation definition
- action definition
- reward specification
- episode lifecycle
- evaluation signals

This layer is where the project becomes trainable. It should be kept modular so that different algorithmic ideas can be tested without rewriting the simulation assets.

### 3. Learning and Control Layer

This layer will eventually contain the actual policy learning and control logic.

Target directions include:

- flat reinforcement learning baselines
- hierarchical reinforcement learning
- skill libraries
- goal-conditioned control
- policy switching between sub-behaviors

In the long term, this layer should support both low-level control policies and higher-level decision modules.

### 4. Human and Multimodal Interaction Layer

This layer is not central to the first environment prototype, but it is a core part of the project vision.

Target directions include:

- visual observations
- language-conditioned task descriptions
- human corrective feedback
- human preference guidance
- vision-language-action style integration

This layer is where the project moves beyond classical robotic control toward more general embodied intelligence.

## Why a Layered Design

A layered design is important for three reasons.

### Reusability

The same environment should be usable with different algorithms, training styles, and observation modalities.

### Interpretability

Each layer should have a clear role. This makes it easier to diagnose whether a failure comes from perception, reward design, planning, or motor execution.

### Extensibility

The current grasping task is intentionally small. The architecture should make it possible to scale toward longer-horizon manipulation without redesigning the project from scratch.

## Current Design Position

At the moment, the project is still close to the simulation and environment-interface layers. This is expected.

The main design achievement so far is not algorithmic novelty. It is the establishment of a clean starting point:

- a robotic manipulation scenario
- a learning-oriented environment wrapper
- a structure that can absorb more advanced learning components later

## Expected Evolution

The design is expected to evolve in roughly this order:

1. stabilize the current environment and benchmark it with standard RL baselines
2. expand the task from simple grasping toward reusable manipulation skills
3. introduce hierarchical control structure
4. integrate vision as a primary source of task-relevant information
5. explore language and human-guided learning loops

## Design Risks

There are several design risks that should be kept in mind.

- The project could become too tightly coupled to a single task if task abstractions are not introduced early.
- Reward design may become brittle if it is too specific to a single scripted success path.
- Hierarchical control may add complexity before a strong flat baseline is established.
- Vision integration can overwhelm the learning problem if added before the control interface is stable.

These risks suggest a disciplined progression: strong environment first, strong baseline second, complexity third.

## Long-Term Goal

The long-term design goal is to support research on robotic arm systems that do more than execute control. The aim is to support systems that can:

- perceive
- plan
- select skills
- adapt
- accept guidance
- generalize across tasks

That is the broader design intention behind this repository.
