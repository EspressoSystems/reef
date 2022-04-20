// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Reef library.

// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Traits describing the interface of a CAP ledger.
use crate::types::{AuditError, AuditMemoOpening};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jf_cap::{
    keys::{AuditorKeyPair, AuditorPubKey},
    proof::UniversalParam,
    structs::{AssetCode, AssetDefinition, Nullifier, RecordCommitment, RecordOpening},
    TransactionNote,
};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// A set of nullifiers which have been spent.
///
/// In its simplest form this is just a set of nullifiers; however, notice that the only required
/// interface is for inserting new nullifiers. Querying nullifiers is done on a network-specific
/// basis, and we do not assume that the [NullifierSet] data structure will provide an interface for
/// querying every nullifier that has been published. This could, for example, be implemented as a
/// cryptographic commitment to a set of nullifiers, where `multi_insert` simply updates the
/// commitment.
///
/// Since [NullifierSet] may be a commitment to a set that is actually represented elsewhere, it
/// includes a notion of a [Proof](NullifierSet::Proof) which can be used to authenticate that a
/// particular nullifier is or isn't in the set. All interfaces involving [NullifierSet] are
/// written with the assumption that proofs are always necessary for inserting nullifiers, and must
/// always be returned as authentication when querying nullifiers. An implementation of this trait
/// which does not require authentication (for example, an implementation that maintains the
/// entire nullifiers set in memory) may use trivial proofs, like `()`.
pub trait NullifierSet:
    Clone + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync
{
    /// Authentication that a given nullifier is or isn't in the set.
    type Proof: Clone + Debug + Send + Sync;

    /// Update the set to include additional nullifiers.
    ///
    /// Insert a collection of nullifiers into the set given proofs that the nullifiers are not
    /// already in the set. If this function fails, it returns one of the input proofs which was
    /// invalid.
    fn multi_insert(&mut self, nullifiers: &[(Nullifier, Self::Proof)]) -> Result<(), Self::Proof>;
}

/// The types of transactions supported by this network.
///
/// This trait represents an enum-like interface, with a variant for each transaction. Thus, the
/// public interface consists of constructors for the CAP transaction type variants. The
/// implementation may include additional variants for extensions to the protocol. It is required
/// that two transaction kinds representing different kinds of transactiosn compare unequal. For
/// example, `TransactionKind::send() != TransactionKind::mint()`.
pub trait TransactionKind:
    Clone + Debug + Display + PartialEq + Eq + Hash + Serialize + DeserializeOwned + Send + Sync
{
    fn send() -> Self;
    fn receive() -> Self;
    fn mint() -> Self;
    fn freeze() -> Self;
    fn unfreeze() -> Self;
    fn unknown() -> Self;
}

/// A CAP transaction.
///
/// A CAP transaction contains a list of input nullifiers and output record commitments, as well as
/// viewing information encrypted under the viewing key of the transaction's asset type. It may
/// contain additional information, such as a proof that the transaction is valid and optional
/// plaintext record openings for the transaction outputs.
pub trait Transaction: Clone + Debug + Serialize + DeserializeOwned + Send + Sync {
    /// Nullifier set to be updated when a transaction is added to the ledger.
    type NullifierSet: NullifierSet;

    /// Transaction digest.
    ///
    /// This should be a determinstic, injective, committing function of the transaction.
    type Hash: Clone
        + Debug
        + Eq
        + Hash
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + CanonicalSerialize
        + CanonicalDeserialize;

    /// Supported transaction types.
    type Kind: TransactionKind;

    /// Wrap a raw CAP transaction in this network's transaction type.
    fn cap(note: TransactionNote, proofs: Vec<<Self::NullifierSet as NullifierSet>::Proof>)
        -> Self;

    /// Attempt to decrypt the attached viewing memo.
    ///
    /// Given a collection of asset types for which the caller holds the viewing key, attempt to
    /// open the viewing memos attached to this transaction.
    ///
    /// `viewable_assets` should be the set of asset types which the caller
    /// can view. This determines which asset types can be viewed by this
    /// method. `viewing_keys` is the caller's collection of viewing key
    /// pairs, indexed by public key. `viewing_keys` must contain every public
    /// key which is listed as a viewer in the policy of one of the
    /// `viewable_assets`.
    fn open_audit_memo(
        &self,
        viewable_assets: &[AssetDefinition],
        viewing_keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    ) -> Result<AuditMemoOpening, AuditError>;

    /// This transaction's input nullifiers.
    ///
    /// The results should contain authentication that the nullifiers were unspent at the time the
    /// transaction was constructed.
    fn proven_nullifiers(&self) -> Vec<(Nullifier, <Self::NullifierSet as NullifierSet>::Proof)>;

    /// Commitments to the records created by this transaction.
    fn output_commitments(&self) -> Vec<RecordCommitment>;

    /// If this is not a private transaction, get the openings of its output records.
    fn output_openings(&self) -> Option<Vec<RecordOpening>> {
        // Most transactions do not have attached record openings. Override this default if the
        // implementing transaction type does.
        None
    }

    /// A committing hash of this transaction.
    fn hash(&self) -> Self::Hash;

    /// The type of this transaction.
    fn kind(&self) -> Self::Kind;

    /// Update the proofs that this transaction's nullifiers are unspent.
    fn set_proofs(&mut self, proofs: Vec<<Self::NullifierSet as NullifierSet>::Proof>);

    /// The number of outputs.
    fn output_len(&self) -> usize {
        // Override with a more efficient implementation if the output length can be calculated
        // without building the vector of outputs.
        self.output_commitments().len()
    }

    /// This transaction's inputs, without authentication.
    fn input_nullifiers(&self) -> Vec<Nullifier> {
        self.proven_nullifiers()
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }
}

/// Errors that can occur when validating a [Transaction] or [Block].
pub trait ValidationError:
    'static + Clone + Debug + Display + snafu::Error + Serialize + DeserializeOwned + Send + Sync
{
    /// A catch-all error variant with a helpful message.
    fn new(msg: impl Display) -> Self;

    /// Whether validation failed because a transaction's nullifier proof was invalid.
    fn is_bad_nullifier_proof(&self) -> bool;
}

/// A block of transactions.
///
/// From the point of view of this abstraction, a [Block] is just a list of transactions, although
/// the implementation of [Block] may contain additional metadata.
pub trait Block: Clone + Debug + Serialize + DeserializeOwned + Send + Sync {
    /// Transactions in this block.
    type Transaction: Transaction;

    /// Errors that can occur when validation this block.
    type Error: ValidationError;

    /// Create a block from a list of transactions.
    fn new(txns: Vec<Self::Transaction>) -> Self;

    /// Add a new [Transaction] to this block.
    ///
    /// Fails if the transaction would make the block inconsistent, for example if the new
    /// transaction is incompatible with a transaction already in the block.
    fn add_transaction(&mut self, txn: Self::Transaction) -> Result<(), Self::Error>;

    /// The transactions in this block.
    fn txns(&self) -> Vec<Self::Transaction>;

    /// The number of transactions in this block.
    fn len(&self) -> usize {
        self.txns().len()
    }

    /// Whether there are no transactions in this block.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// State required to validate a [Block] of [Transaction]s.
///
/// Technically, this interface does not actually require blocks to be validated. It merely requires
/// that the [Validator] state is sufficient to produce the outputs of each block, and that each
/// state has a unique commitment. If the network as a whole uses commitments to authenticate ledger
/// states, then [Validator::StateCommitment] should match that commitment after each block.
pub trait Validator:
    Clone + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync
{
    /// A commitment to the state of the validator.
    type StateCommitment: Copy + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync;

    /// Blocks applied by this validator.
    type Block: Block;

    /// The number of blocks this validator has applied.
    fn now(&self) -> u64;

    /// The commitment to the current state of the validator.
    fn commit(&self) -> Self::StateCommitment;

    /// Apply a new block, updating the state and returning UIDs for each transaction outputs.
    ///
    /// For each output, the returned UID is the index of that output; that is, the total number of
    /// outputs which had been generated before that one.
    fn validate_and_apply(
        &mut self,
        block: Self::Block,
    ) -> Result<Vec<u64>, <Self::Block as Block>::Error>;
}

/// A CAP ledger.
///
/// This trait aggregates various other traits that constitute a ledger and network. It also
/// provides whole-ledger information.
pub trait Ledger: Copy + Debug + Send + Sync {
    /// The state of a validator for this ledger.
    ///
    /// Note that this determines all of the other types, such as [Block] and [Transaction], used by
    /// this ledger.
    type Validator: Validator;

    /// A human-readable name for this ledger.
    fn name() -> String;

    /// The number of past [MerkleTree](jf_cap::MerkleTree) roots maintained by validators.
    ///
    /// This determines how out of data a CAP transaction can be and still be accepted by a
    /// validator. If a transaction was constructed against a ledger state which is less than
    /// [record_root_history](Ledger::record_root_history) transactions old, the transaction can be
    /// accepted.
    fn record_root_history() -> usize;

    /// The height of the ledger's records [MerkleTree](jf_cap::MerkleTree).
    fn merkle_height() -> u8;

    /// The universal setup for PLONK proofs.
    fn srs() -> &'static UniversalParam;
}
