# Reef
An abstraction of an entire AAP-style network

## Traits and types
This crate defines traits and types for working with an AAP-style network abstractly. It allows the
development of network-agnostic components, by parameterizing such components on a type
`L: reef::traits::Ledger`. These components can then be used with any compliant AAP-style network
which implements these traits.

## Minimal implementation
There is also a minimal implementation of the AAP ledger traits, in terms of Jellyfish AAP
cryptographic primitives. This implementation can be used as a foundation for building more complex
AAP ledgers. It can also be used as an easy mock implementation to test a network-agnostic component
in isolation from any particular network.
