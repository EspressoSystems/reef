//! Minimal implementation of the ledger traits for an AAP-style ledger. This implementation can be
//! used as an aid in implemeneting the traits for more complex, specific ledger types. It is also
//! fully functional and can be used as a mock ledger for testing downstream modules that are
//! parameterized by ledger type.

use crate::traits;
use crate::types::{AuditError, AuditMemoOpening};
use arbitrary::Arbitrary;
use commit::{Commitment, Committable};
use jf_aap::{
    keys::{AuditorKeyPair, AuditorPubKey},
    mint::MintNote,
    structs::{AssetCode, AssetDefinition, Nullifier, RecordCommitment},
    transfer::TransferNote,
    TransactionNote,
};
use serde::{Deserialize, Serialize};
use snafu::Snafu;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub type NullifierSet = HashSet<Nullifier>;

impl traits::NullifierSet for NullifierSet {
    type Proof = ();

    fn multi_insert(&mut self, nullifiers: &[(Nullifier, Self::Proof)]) -> Result<(), Self::Proof> {
        for (n, _) in nullifiers {
            self.insert(*n);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, strum_macros::Display)]
pub enum TransactionKind {
    Mint,
    Freeze,
    Unfreeze,
    Send,
    Receive,
    Unknown,
}

impl traits::TransactionKind for TransactionKind {
    fn send() -> Self {
        Self::Send
    }

    fn receive() -> Self {
        Self::Receive
    }

    fn mint() -> Self {
        Self::Mint
    }

    fn freeze() -> Self {
        Self::Freeze
    }

    fn unfreeze() -> Self {
        Self::Unfreeze
    }

    fn unknown() -> Self {
        Self::Unknown
    }
}

pub type Transaction = TransactionNote;

impl traits::Transaction for TransactionNote {
    type NullifierSet = NullifierSet;
    type Hash = Commitment<Self>;
    type Kind = TransactionKind;

    fn aap(note: TransactionNote, _proofs: Vec<()>) -> Self {
        note
    }

    fn open_audit_memo(
        &self,
        assets: &HashMap<AssetCode, AssetDefinition>,
        keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    ) -> Result<AuditMemoOpening, AuditError> {
        match self {
            Self::Transfer(xfr) => open_xfr_audit_memo(assets, keys, xfr),
            Self::Mint(mint) => open_mint_audit_memo(keys, mint),
            Self::Freeze(_) => Err(AuditError::NoAuditMemos),
        }
    }

    fn proven_nullifiers(&self) -> Vec<(Nullifier, ())> {
        TransactionNote::nullifiers(self)
            .into_iter()
            .map(|n| (n, ()))
            .collect()
    }

    fn output_commitments(&self) -> Vec<RecordCommitment> {
        TransactionNote::output_commitments(self)
    }

    fn hash(&self) -> Self::Hash {
        self.commit()
    }

    fn kind(&self) -> Self::Kind {
        match self {
            Self::Transfer(..) => TransactionKind::Send,
            Self::Mint(..) => TransactionKind::Mint,
            Self::Freeze(..) => TransactionKind::Freeze,
        }
    }

    fn set_proofs(&mut self, _proofs: Vec<()>) {
        // Our nullifier proofs are trivial, so there's nothing to do.
    }
}

pub fn open_xfr_audit_memo(
    assets: &HashMap<AssetCode, AssetDefinition>,
    keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    xfr: &TransferNote,
) -> Result<AuditMemoOpening, AuditError> {
    for asset in assets.values() {
        let audit_key = &keys[asset.policy_ref().auditor_pub_key()];
        if let Ok((inputs, outputs)) = audit_key.open_transfer_audit_memo(asset, xfr) {
            return Ok(AuditMemoOpening {
                asset: asset.clone(),
                inputs,
                outputs,
            });
        }
    }
    Err(AuditError::UnauditableAsset)
}

pub fn open_mint_audit_memo(
    keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    mint: &MintNote,
) -> Result<AuditMemoOpening, AuditError> {
    keys.get(mint.mint_asset_def.policy_ref().auditor_pub_key())
        .ok_or(AuditError::UnauditableAsset)
        .map(|audit_key| {
            let output = audit_key.open_mint_audit_memo(mint).unwrap();
            AuditMemoOpening {
                asset: mint.mint_asset_def.clone(),
                inputs: vec![],
                outputs: vec![output],
            }
        })
}

#[derive(Clone, Debug, Serialize, Deserialize, Snafu)]
pub enum ValidationError {
    Failed { msg: String },
}

impl traits::ValidationError for ValidationError {
    fn new(msg: impl Display) -> Self {
        Self::Failed {
            msg: msg.to_string(),
        }
    }

    fn is_bad_nullifier_proof(&self) -> bool {
        false
    }
}

pub type Block = Vec<Transaction>;

impl traits::Block for Block {
    type Transaction = Transaction;
    type Error = ValidationError;

    fn new(txns: Vec<Transaction>) -> Self {
        txns
    }

    fn add_transaction(&mut self, txn: Transaction) -> Result<(), Self::Error> {
        self.push(txn);
        Ok(())
    }

    fn txns(&self) -> Vec<Transaction> {
        self.clone()
    }
}

// Our minimal validator will not actually validate. It only does the least it can do to implement
// the Validator interface, namely
//  * compute a unique commitment after each block (this is just the count of blocks)
//  * compute the UIDs for the outputs of each block (by counting the number of outputs total)
#[derive(Arbitrary, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Validator {
    pub now: u64,
    pub num_records: u64,
}

impl traits::Validator for Validator {
    type StateCommitment = u64;
    type Block = Block;

    fn now(&self) -> u64 {
        self.now
    }

    fn commit(&self) -> Self::StateCommitment {
        self.now
    }

    fn validate_and_apply(&mut self, block: Self::Block) -> Result<Vec<u64>, ValidationError> {
        let mut uids = vec![];
        let mut uid = self.num_records;
        for txn in block {
            for _ in 0..txn.output_len() {
                uids.push(uid);
                uid += 1;
            }
        }
        self.num_records = uid;
        self.now += 1;

        Ok(uids)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ledger;

impl traits::Ledger for Ledger {
    type Validator = Validator;

    fn name() -> String {
        String::from("Minimal AAP Ledger")
    }

    fn merkle_height() -> u8 {
        5
    }

    fn record_root_history() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Ledger as _, NullifierSet as _, Transaction as _, Validator as _};
    use jf_aap::{
        freeze::{FreezeNote, FreezeNoteInput},
        keys::{FreezerKeyPair, UserKeyPair},
        proof::{universal_setup, UniversalParam},
        structs::{
            AssetPolicy, FeeInput, FreezeFlag, NoteType, RecordCommitment, RecordOpening,
            TxnFeeInfo,
        },
        transfer::TransferNoteInput,
        utils::compute_universal_param_size,
        AccMemberWitness, MerkleTree,
    };
    use lazy_static::lazy_static;
    use rand_chacha::{rand_core::SeedableRng, ChaChaRng};

    lazy_static! {
        static ref UNIVERSAL_PARAM: UniversalParam = universal_setup(
            *[
                compute_universal_param_size(NoteType::Transfer, 3, 3, Ledger::merkle_height())
                    .unwrap_or_else(|err| {
                        panic!(
                            "Error while computing the universal parameter size for Transfer: {}",
                            err
                        )
                    },),
                compute_universal_param_size(NoteType::Mint, 0, 0, Ledger::merkle_height())
                    .unwrap_or_else(|err| {
                        panic!(
                            "Error while computing the universal parameter size for Mint: {}",
                            err
                        )
                    },),
                compute_universal_param_size(NoteType::Freeze, 2, 2, Ledger::merkle_height())
                    .unwrap_or_else(|err| {
                        panic!(
                            "Error while computing the universal parameter size for Freeze: {}",
                            err
                        )
                    },),
            ]
            .iter()
            .max()
            .unwrap(),
            &mut ChaChaRng::from_seed([42u8; 32])
        )
        .unwrap();
    }

    #[test]
    fn test_nullifier_set() {
        let mut rng = ChaChaRng::from_seed([42u8; 32]);

        let mut s = NullifierSet::default();
        assert_eq!(s.len(), 0);

        let n = Nullifier::random_for_test(&mut rng);
        s.multi_insert(&[(n, ())]).unwrap();
        assert_eq!(s.len(), 1);
        assert!(s.contains(&n));
    }

    #[test]
    fn test_transaction() {
        let mut rng = ChaChaRng::from_seed([42u8; 32]);
        let key = UserKeyPair::generate(&mut rng);
        let freezer_key = FreezerKeyPair::generate(&mut rng);
        let auditor_key = AuditorKeyPair::generate(&mut rng);

        let xfr_proving_key =
            jf_aap::proof::transfer::preprocess(&*UNIVERSAL_PARAM, 2, 2, Ledger::merkle_height())
                .unwrap()
                .0;
        let mint_proving_key =
            jf_aap::proof::mint::preprocess(&*UNIVERSAL_PARAM, Ledger::merkle_height())
                .unwrap()
                .0;
        let freeze_proving_key =
            jf_aap::proof::freeze::preprocess(&*UNIVERSAL_PARAM, 2, Ledger::merkle_height())
                .unwrap()
                .0;

        // Set up a ledger. For simplicity we will use the same ledger state and fee input for each
        // transaction (mint, transfer, and freeze);
        let mut records = MerkleTree::new(Ledger::merkle_height()).unwrap();
        let fee_ro = RecordOpening::new(
            &mut rng,
            1,
            AssetDefinition::native(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let fee_comm = RecordCommitment::from(&fee_ro);
        records.push(fee_comm.to_field_element());
        let fee_nullifier = key.nullify(
            AssetDefinition::native().policy_ref().freezer_pub_key(),
            0,
            &fee_comm,
        );

        // To freeze and audit, we need a record of a non-native asset type.
        let (asset_code, seed) = AssetCode::random(&mut rng);
        let policy = AssetPolicy::default()
            .set_freezer_pub_key(freezer_key.pub_key())
            .set_auditor_pub_key(auditor_key.pub_key())
            .reveal_user_address()
            .unwrap()
            .reveal_amount()
            .unwrap();
        let asset_def = AssetDefinition::new(asset_code, policy).unwrap();
        let asset_ro = RecordOpening::new(
            &mut rng,
            1,
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let asset_comm = RecordCommitment::from(&asset_ro);
        records.push(asset_comm.to_field_element());
        let asset_nullifier = key.nullify(&freezer_key.pub_key(), 1, &asset_comm);

        // Now `records` is in the state we are going to use for the tests. We can look up Merkle
        // proofs as needed.
        let fee_input = FeeInput {
            ro: fee_ro.clone(),
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 0)
                .expect_ok()
                .unwrap()
                .1,
            owner_keypair: &key,
        };

        // Generate one transaction of each type.
        let mint_ro = RecordOpening::new(
            &mut rng,
            1,
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let mint_comm = RecordCommitment::from(&mint_ro);
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1).unwrap().0;
        let mint_note = MintNote::generate(
            &mut rng,
            mint_ro.clone(),
            seed,
            &[],
            fee_info,
            &mint_proving_key,
        )
        .unwrap()
        .0;
        let mint = TransactionNote::Mint(Box::new(mint_note.clone()));

        let xfr_inputs = vec![TransferNoteInput {
            ro: asset_ro.clone(),
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 1)
                .expect_ok()
                .unwrap()
                .1,
            owner_keypair: &key,
            cred: None,
        }];
        let xfr_ro = RecordOpening::new(
            &mut rng,
            1,
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let xfr_comm = RecordCommitment::from(&xfr_ro);
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1).unwrap().0;
        let xfr_note = TransferNote::generate_non_native(
            &mut rng,
            xfr_inputs,
            &[xfr_ro],
            fee_info,
            2u64.pow(jf_aap::constants::MAX_TIMESTAMP_LEN as u32) - 1,
            &xfr_proving_key,
            vec![],
        )
        .unwrap()
        .0;
        let xfr = TransactionNote::Transfer(Box::new(xfr_note.clone()));

        // To freeze, we need a record of a non-native asset type.
        let freeze_inputs = vec![FreezeNoteInput {
            ro: asset_ro.clone(),
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 1)
                .expect_ok()
                .unwrap()
                .1,
            keypair: &freezer_key,
        }];
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1).unwrap().0;
        let freeze_note =
            FreezeNote::generate(&mut rng, freeze_inputs, fee_info, &freeze_proving_key)
                .unwrap()
                .0;
        let freeze = TransactionNote::Freeze(Box::new(freeze_note.clone()));

        // Test Transaction interface.
        assert_eq!(mint.kind(), TransactionKind::Mint);
        assert_eq!(mint.output_len(), 2);
        assert_eq!(
            mint.output_commitments(),
            vec![mint_note.chg_comm, mint_comm]
        );
        assert_eq!(mint.input_nullifiers(), vec![fee_nullifier]);

        assert_eq!(xfr.kind(), TransactionKind::Send);
        assert_eq!(xfr.output_len(), 2);
        assert_eq!(xfr.output_commitments(), xfr_note.output_commitments);
        assert_eq!(xfr.output_commitments()[1], xfr_comm);
        assert_eq!(xfr.input_nullifiers(), vec![fee_nullifier, asset_nullifier]);

        assert_eq!(freeze.kind(), TransactionKind::Freeze);
        assert_eq!(freeze.output_len(), 2);
        assert_eq!(freeze.output_commitments(), freeze_note.output_commitments);
        assert_eq!(
            freeze.input_nullifiers(),
            vec![
                fee_nullifier,
                key.nullify(&freezer_key.pub_key(), 1, &asset_comm)
            ]
        );

        // Check that each transaction has a different hash.
        assert_ne!(
            traits::Transaction::hash(&mint),
            traits::Transaction::hash(&xfr)
        );
        assert_ne!(
            traits::Transaction::hash(&xfr),
            traits::Transaction::hash(&freeze)
        );
        assert_ne!(
            traits::Transaction::hash(&freeze),
            traits::Transaction::hash(&mint)
        );

        // Check auditing interface.
        let auditable_assets = vec![(asset_code, asset_def.clone())].into_iter().collect();
        let audit_keys = vec![(auditor_key.pub_key(), auditor_key)]
            .into_iter()
            .collect();

        let mint_memo = mint
            .open_audit_memo(&auditable_assets, &audit_keys)
            .unwrap();
        assert_eq!(mint_memo.asset, asset_def);
        assert_eq!(mint_memo.inputs, vec![]);
        assert_eq!(mint_memo.outputs.len(), 1);
        assert_eq!(mint_memo.outputs[0].asset_code, asset_code);
        assert_eq!(mint_memo.outputs[0].user_address, Some(key.address()));
        assert_eq!(mint_memo.outputs[0].amount, Some(1));
        let xfr_memo = xfr.open_audit_memo(&auditable_assets, &audit_keys).unwrap();
        assert_eq!(xfr_memo.asset, asset_def);
        assert_eq!(xfr_memo.inputs.len(), 1);
        assert_eq!(xfr_memo.inputs[0].asset_code, asset_code);
        assert_eq!(xfr_memo.inputs[0].user_address, Some(key.address()));
        assert_eq!(xfr_memo.inputs[0].amount, Some(1));
        assert_eq!(xfr_memo.outputs.len(), 1);
        assert_eq!(xfr_memo.outputs[0].asset_code, asset_code);
        assert_eq!(xfr_memo.outputs[0].user_address, Some(key.address()));
        assert_eq!(xfr_memo.outputs[0].amount, Some(1));

        assert_eq!(
            freeze.open_audit_memo(&auditable_assets, &audit_keys),
            Err(AuditError::NoAuditMemos)
        );
    }

    #[test]
    fn test_validator() {
        let mut validator = Validator::default();
        assert_eq!(validator.now(), 0);
        let initial_commit = validator.commit();

        // Build a block to apply. For this we need a transaction note. It doesn't have to be valid, but
        // the easiest way to build a transaction note requires it to be valid, so we'll set up a whole
        // mock ledger.
        let mut rng = ChaChaRng::from_seed([42u8; 32]);
        let key = UserKeyPair::generate(&mut rng);
        let mint_proving_key =
            jf_aap::proof::mint::preprocess(&*UNIVERSAL_PARAM, Ledger::merkle_height())
                .unwrap()
                .0;
        let mut records = MerkleTree::new(Ledger::merkle_height()).unwrap();
        let fee_ro = RecordOpening::new(
            &mut rng,
            1,
            AssetDefinition::native(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let fee_comm = RecordCommitment::from(&fee_ro);
        records.push(fee_comm.to_field_element());
        let fee_input = FeeInput {
            ro: fee_ro.clone(),
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 0)
                .expect_ok()
                .unwrap()
                .1,
            owner_keypair: &key,
        };

        // Generate a mint transaction.
        let (asset_code, seed) = AssetCode::random(&mut rng);
        let asset_def = AssetDefinition::new(asset_code, AssetPolicy::default()).unwrap();
        let mint_ro = RecordOpening::new(
            &mut rng,
            1,
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1).unwrap().0;
        let mint_note = MintNote::generate(
            &mut rng,
            mint_ro.clone(),
            seed,
            &[],
            fee_info,
            &mint_proving_key,
        )
        .unwrap()
        .0;
        let mint = TransactionNote::Mint(Box::new(mint_note.clone()));

        // Apply a block and check that the correct UIDs are computed.
        assert_eq!(
            validator.validate_and_apply(vec![mint.clone()]).unwrap(),
            vec![0, 1]
        );
        // Make sure we have a new timestamp and commit.
        assert_eq!(validator.now(), 1);
        assert_ne!(validator.commit(), initial_commit);

        // Apply another block and check that we get different UIDs. Technically it's not allowed to
        // apply the same block twice, but our minimal validator doesn't care.
        assert_eq!(
            validator.validate_and_apply(vec![mint]).unwrap(),
            vec![2, 3]
        );
    }
}
