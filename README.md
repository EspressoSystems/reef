# Reef
An abstraction of an entire CAP-style network

## Traits and types
This crate defines traits and types for working with an CAP-style network abstractly. It allows the
development of network-agnostic components, by parameterizing such components on a type
`L: reef::traits::Ledger`. These components can then be used with any compliant CAP-style network
which implements these traits.

## Minimal implementation
There is also a minimal implementation of the CAP ledger traits, in terms of Jellyfish CAP
cryptographic primitives. This implementation can be used as a foundation for building more complex
CAP ledgers. It can also be used as an easy mock implementation to test a network-agnostic component
in isolation from any particular network.
