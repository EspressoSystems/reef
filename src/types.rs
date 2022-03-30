// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Reef library.

// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Ledger types
//!
//! This module contains type definitions, and type aliases for commonly used associated types from
//! the [traits] module.

use crate::traits;
use jf_cap::structs::{AssetDefinition, AuditData};

/// A validator for a ledger `L`.
pub type Validator<L> = <L as traits::Ledger>::Validator;
/// A commitment to a validator state for a ledger `L`.
pub type StateCommitment<L> = <Validator<L> as traits::Validator>::StateCommitment;
/// A block of transactions that can be applied to a ledger `L`.
pub type Block<L> = <Validator<L> as traits::Validator>::Block;
/// An error that can occur while state validating transitions of a ledger `L`.
pub type ValidationError<L> = <Block<L> as traits::Block>::Error;
/// A transaction that can be applied to a ledger `L`.
pub type Transaction<L> = <Block<L> as traits::Block>::Transaction;
/// A committing hash of a transaction that can be applied to a ledger `L`.
pub type TransactionHash<L> = <Transaction<L> as traits::Transaction>::Hash;
/// Types of transactions supported by a ledger `L`.
pub type TransactionKind<L> = <Transaction<L> as traits::Transaction>::Kind;
/// A set of spent nullifiers for a ledger `L`.
pub type NullifierSet<L> = <Transaction<L> as traits::Transaction>::NullifierSet;
/// A proof that a nullifier is spent or unspent, relative to a ledger `L`.
pub type NullifierProof<L> = <NullifierSet<L> as traits::NullifierSet>::Proof;

/// Information contained in a viewing memo.
#[derive(Clone, Debug, PartialEq)]
pub struct AuditMemoOpening {
    pub asset: AssetDefinition,
    pub inputs: Vec<AuditData>,
    pub outputs: Vec<AuditData>,
}

/// Errors that can occur when trying to decrypt a viewing memo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuditError {
    UnauditableAsset,
    NoAuditMemos,
}
