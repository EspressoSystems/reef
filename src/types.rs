// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Reef library.

// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

use crate::traits;
use jf_cap::structs::{AssetDefinition, AuditData};

pub type Validator<L> = <L as traits::Ledger>::Validator;
pub type StateCommitment<L> = <Validator<L> as traits::Validator>::StateCommitment;
pub type Block<L> = <Validator<L> as traits::Validator>::Block;
pub type ValidationError<L> = <Block<L> as traits::Block>::Error;
pub type Transaction<L> = <Block<L> as traits::Block>::Transaction;
pub type TransactionHash<L> = <Transaction<L> as traits::Transaction>::Hash;
pub type TransactionKind<L> = <Transaction<L> as traits::Transaction>::Kind;
pub type NullifierSet<L> = <Transaction<L> as traits::Transaction>::NullifierSet;
pub type NullifierProof<L> = <NullifierSet<L> as traits::NullifierSet>::Proof;

#[derive(Clone, Debug, PartialEq)]
pub struct AuditMemoOpening {
    pub asset: AssetDefinition,
    pub inputs: Vec<AuditData>,
    pub outputs: Vec<AuditData>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuditError {
    UnauditableAsset,
    NoAuditMemos,
}
