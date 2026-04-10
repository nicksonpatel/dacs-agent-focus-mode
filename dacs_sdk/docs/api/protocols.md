# Protocols

Core data types used throughout DACS.

## Enums

### AgentStatus

::: dacs._protocols.AgentStatus

### UrgencyLevel

::: dacs._protocols.UrgencyLevel

## Dataclasses

### SteeringRequest

::: dacs._protocols.SteeringRequest

### SteeringResponse

::: dacs._protocols.SteeringResponse

### FocusContext

::: dacs._protocols.FocusContext

### RegistryEntry

::: dacs._protocols.RegistryEntry

### RegistryUpdate

::: dacs._protocols.RegistryUpdate

## SteeringRequestQueue

::: dacs._protocols.SteeringRequestQueue
    options:
      show_source: true
      members:
        - __init__
        - enqueue
        - dequeue
        - peek
        - has_high_urgency
        - __len__
