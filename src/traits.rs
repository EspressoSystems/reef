use crate::types::{AuditError, AuditMemoOpening};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jf_aap::{
    keys::{AuditorKeyPair, AuditorPubKey},
    structs::{AssetCode, AssetDefinition, Nullifier, RecordCommitment, RecordOpening},
    TransactionNote,
};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub trait NullifierSet:
    Clone + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync
{
    type Proof: Clone + Debug + Send + Sync;

    // Insert a collection of nullifiers into the set, given proofs that the nullifiers are not
    // already in the set. If this function fails, it returns one of the input proofs which was
    // invalid.
    fn multi_insert(&mut self, nullifiers: &[(Nullifier, Self::Proof)]) -> Result<(), Self::Proof>;
}

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

pub trait Transaction: Clone + Debug + Serialize + DeserializeOwned + Send + Sync {
    type NullifierSet: NullifierSet;
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
    type Kind: TransactionKind;

    fn aap(note: TransactionNote, proofs: Vec<<Self::NullifierSet as NullifierSet>::Proof>)
        -> Self;

    // Given a collection of asset types that the caller is able to audit, attempt to open the
    // audit memos attached to this transaction.
    //
    // `auditable_assets` should be the set of asset types which the caller can audit, indexed
    // by asset code. This determines which asset types can be audited by this method.
    // `auditor_keys` is the caller's collection of auditing key pairs, indexed by public key.
    // `auditor_keys` must contain every public key which is listed as an auditor in the policy
    // of one of the `auditable_assets`.
    fn open_audit_memo(
        &self,
        auditable_assets: &HashMap<AssetCode, AssetDefinition>,
        auditor_keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    ) -> Result<AuditMemoOpening, AuditError>;
    fn proven_nullifiers(&self) -> Vec<(Nullifier, <Self::NullifierSet as NullifierSet>::Proof)>;
    fn output_commitments(&self) -> Vec<RecordCommitment>;
    // Tries to get record openings corresponding to the outputs of this transaction. If
    // possible, the wallet should add any relevant openings right away when this transaction is
    // received. Otherwise, it will wait for corresponding receiver memos.
    fn output_openings(&self) -> Option<Vec<RecordOpening>> {
        // Most transactions do not have attached record openings. Override this default if the
        // implementing transaction type does.
        None
    }
    fn hash(&self) -> Self::Hash;
    fn kind(&self) -> Self::Kind;

    fn set_proofs(&mut self, proofs: Vec<<Self::NullifierSet as NullifierSet>::Proof>);

    // Override with a more efficient implementation if the output length can be calculated
    // without building the vector of outputs.
    fn output_len(&self) -> usize {
        self.output_commitments().len()
    }

    fn input_nullifiers(&self) -> Vec<Nullifier> {
        self.proven_nullifiers()
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }
}

pub trait ValidationError:
    'static + Clone + Debug + Display + snafu::Error + Serialize + DeserializeOwned + Send + Sync
{
    fn new(msg: impl Display) -> Self;
    fn is_bad_nullifier_proof(&self) -> bool;
}

pub trait Block: Clone + Debug + Serialize + DeserializeOwned + Send + Sync {
    type Transaction: Transaction;
    type Error: ValidationError;
    fn new(txns: Vec<Self::Transaction>) -> Self;
    fn add_transaction(&mut self, txn: Self::Transaction) -> Result<(), Self::Error>;
    fn txns(&self) -> Vec<Self::Transaction>;
    fn len(&self) -> usize {
        self.txns().len()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Validator:
    Clone + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync
{
    type StateCommitment: Copy + Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync;
    type Block: Block;

    fn now(&self) -> u64;
    fn commit(&self) -> Self::StateCommitment;
    fn validate_and_apply(
        &mut self,
        block: Self::Block,
    ) -> Result<Vec<u64>, <Self::Block as Block>::Error>;
}

pub trait Ledger: Copy + Debug + Send + Sync {
    type Validator: Validator;
    fn name() -> String;
}
