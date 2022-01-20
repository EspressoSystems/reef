use crate::traits;
use jf_aap::structs::{AssetDefinition, AuditData};

pub type Validator<L> = <L as traits::Ledger>::Validator;
pub type StateCommitment<L> = <Validator<L> as traits::Validator>::StateCommitment;
pub type Block<L> = <Validator<L> as traits::Validator>::Block;
pub type ValidationError<L> = <Block<L> as traits::Block>::Error;
pub type Transaction<L> = <Block<L> as traits::Block>::Transaction;
pub type TransactionHash<L> = <Transaction<L> as traits::Transaction>::Hash;
pub type TransactionKind<L> = <Transaction<L> as traits::Transaction>::Kind;
pub type NullifierSet<L> = <Transaction<L> as traits::Transaction>::NullifierSet;
pub type NullifierProof<L> = <NullifierSet<L> as traits::NullifierSet>::Proof;

pub struct AuditMemoOpening {
    pub asset: AssetDefinition,
    pub inputs: Vec<AuditData>,
    pub outputs: Vec<AuditData>,
}

pub enum AuditError {
    UnauditableAsset,
    NoAuditMemos,
}
