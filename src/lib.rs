// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Reef library.

// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! An abstraction of a CAP-style network.
//!
//! A network is CAP-style if it implements a UTXO-based blockchain where transactions are defined
//! by nullifiers of the records they consume and commitments to the records they create. The
//! network must support mint, transfer, and freeze transactions, as well as configurable asset
//! privacy with viewing and freezing keys. It may optionally support additional operations that
//! extend the CAP protocol. This library makes no assumptions about the detailed behavior of the
//! network, such as the network architecture or the validation algorithm. It merely provides the
//! types and traits needed by clients to interface with a CAP network.
//!
//! ## Traits and types
//! This crate defines [traits] and [types] for working with a CAP-style network abstractly. It
//! allows the development of network-agnostic components, by parameterizing such components on a
//! type `L` which implements [Ledger]. These components can then be used with any compliant
//! CAP-style network which implements these traits.
//!
//! ## Minimal implementation
//! There is also a minimal implementation of the CAP ledger traits, in terms of Jellyfish CAP
//! cryptographic primitives. The [cap] implementation can be used as a foundation for building
//! more complex CAP ledgers. It can also be used as an easy mock implementation to test a
//! network-agnostic component in isolation from any particular network.

pub mod cap;
pub mod traits;
pub mod types;

pub use traits::Ledger;
pub use types::*;
