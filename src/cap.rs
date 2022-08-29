// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Reef library.

// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Minimal implementation of the ledger traits for a CAP-style ledger.
//!
//! This implementation can be used as an aid in implemeneting the traits for more complex, specific
//! ledger types. It is also fully functional and can be used as a mock ledger for testing
//! downstream modules that are parameterized by ledger type.

use crate::traits;
use crate::types::{ViewingError, ViewingMemoOpening};
use arbitrary::{Arbitrary, Unstructured};
use arbitrary_wrappers::{ArbitraryMerkleTree, ArbitraryNullifier};
use commit::{Commitment, Committable};
use jf_cap::{
    keys::{ViewerKeyPair, ViewerPubKey},
    mint::MintNote,
    proof::UniversalParam,
    structs::{AssetCode, AssetDefinition, Nullifier, RecordCommitment},
    transfer::TransferNote,
    MerkleCommitment, MerkleFrontier, MerkleTree, TransactionNote,
};
use jf_primitives::merkle_tree::FilledMTBuilder;
use lazy_static::lazy_static;
use rand_chacha::{rand_core::SeedableRng, ChaChaRng};
use serde::{Deserialize, Serialize};
use snafu::Snafu;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// A set of nullifiers.
///
/// The simplest implementation of the CAP [NullifierSet](traits::NullifierSet) trait simply stores
/// all the nullifiers that are inserted in a [HashSet].
pub type NullifierSet = HashSet<ArbitraryNullifier>;

impl traits::NullifierSet for NullifierSet {
    type Proof = ();

    fn multi_insert(&mut self, nullifiers: &[(Nullifier, Self::Proof)]) -> Result<(), Self::Proof> {
        for (n, _) in nullifiers {
            self.insert((*n).into());
        }
        Ok(())
    }
}

/// All the kinds of transactions in the basic CAP protocol.
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

/// A CAP transaction.
pub type Transaction = TransactionNote;

impl traits::Transaction for TransactionNote {
    type NullifierSet = NullifierSet;
    type Hash = Commitment<Self>;
    type Kind = TransactionKind;

    fn cap(note: TransactionNote, _proofs: Vec<()>) -> Self {
        note
    }

    fn open_viewing_memo(
        &self,
        assets: &HashMap<AssetCode, AssetDefinition>,
        keys: &HashMap<ViewerPubKey, ViewerKeyPair>,
    ) -> Result<ViewingMemoOpening, ViewingError> {
        match self {
            Self::Transfer(xfr) => open_xfr_viewing_memo(assets, keys, xfr),
            Self::Mint(mint) => open_mint_viewing_memo(keys, mint),
            Self::Freeze(_) => Err(ViewingError::NoViewingMemos),
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

/// Attempt to open the viewer memo attached to a CAP transfer transaction.
///
/// `assets` should be the set of asset types for which the caller holds the viewing key, indexed by
/// asset code. This determines which asset types can be viewed by this method. `keys` is the
/// caller's collection of viewing key pairs, indexed by public key. `keys` must contain every
/// public key which is listed as a viewer in the policy of one of the `assets`.
pub fn open_xfr_viewing_memo(
    assets: &HashMap<AssetCode, AssetDefinition>,
    keys: &HashMap<ViewerPubKey, ViewerKeyPair>,
    xfr: &TransferNote,
) -> Result<ViewingMemoOpening, ViewingError> {
    for asset in assets.values() {
        let viewing_key = &keys[asset.policy_ref().viewer_pub_key()];
        if let Ok((inputs, outputs)) = viewing_key.open_transfer_viewing_memo(asset, xfr) {
            return Ok(ViewingMemoOpening {
                asset: asset.clone(),
                inputs,
                outputs,
            });
        }
    }
    Err(ViewingError::UnviewableAsset)
}

/// Attempt to open the viewer memo attached to a CAP mint transaction.
///
/// `keys` should be the caller's collection of viewing key pairs, indexed by public key.
pub fn open_mint_viewing_memo(
    keys: &HashMap<ViewerPubKey, ViewerKeyPair>,
    mint: &MintNote,
) -> Result<ViewingMemoOpening, ViewingError> {
    keys.get(mint.mint_asset_def.policy_ref().viewer_pub_key())
        .ok_or(ViewingError::UnviewableAsset)
        .map(|viewing_key| {
            let output = viewing_key.open_mint_viewing_memo(mint).unwrap();
            ViewingMemoOpening {
                asset: mint.mint_asset_def.clone(),
                inputs: vec![],
                outputs: vec![output],
            }
        })
}

/// Errors in mock CAP validation.
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

/// A block of CAP transactions.
///
/// The simplest implementation of the [Block](traits::Block) trait is simply a list of CAP
/// transactions.
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

/// A mock CAP validator.
///
/// The minimal validator will not actually validate. It only does the least it can do to implement
/// the [Validator](traits::Validator) interface, namely
///  * compute a unique commitment after each block (this is just the count of blocks)
///  * compute the UIDs and Merkle paths for the outputs of each block
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validator<const H: u8> {
    pub now: u64,
    pub records_commitment: MerkleCommitment,
    pub records_frontier: MerkleFrontier,
}

impl<const H: u8> Default for Validator<H> {
    fn default() -> Self {
        let records = MerkleTree::new(H).unwrap();
        Self {
            now: 0,
            records_commitment: records.commitment(),
            records_frontier: records.frontier(),
        }
    }
}

impl<'a, const H: u8> Arbitrary<'a> for Validator<H> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let records = ArbitraryMerkleTree::arbitrary(u)?.0;
        Ok(Self {
            now: u.arbitrary()?,
            records_commitment: records.commitment(),
            records_frontier: records.frontier(),
        })
    }
}

impl<const H: u8> traits::Validator for Validator<H> {
    type StateCommitment = u64;
    type Block = Block;

    fn now(&self) -> u64 {
        self.now
    }

    fn commit(&self) -> Self::StateCommitment {
        self.now
    }

    fn validate_and_apply(
        &mut self,
        block: Self::Block,
    ) -> Result<(Vec<u64>, MerkleTree), ValidationError> {
        let mut uids = vec![];
        let mut uid = self.records_commitment.num_leaves;
        let mut builder =
            FilledMTBuilder::from_frontier(&self.records_commitment, &self.records_frontier)
                .ok_or_else(|| ValidationError::Failed {
                    msg: "failed to restore Merkle tree from frontier".to_string(),
                })?;
        for txn in block {
            for comm in txn.output_commitments() {
                builder.push(comm.to_field_element());
                uids.push(uid);
                uid += 1;
            }
        }
        let records = builder.build();

        self.now += 1;
        self.records_commitment = records.commitment();
        self.records_frontier = records.frontier();

        Ok((uids, records))
    }
}

/// A minimal CAP ledger.
///
/// The ledger implementation is parameterized on the height of Merkle trees that it will be used
/// with. Test code can use this to set the height very small (e.g. `LedgerWithHeight<5>`) to speed
/// up tests, while code that is trying to closely simulate a production environment can set a more
/// realistic height.
// The `Ledger` implementation includes a constructor for a CAP SRS, so we only enable it in test
// environments or environments where we have a secure construction of the SRS.
#[cfg(any(test, feature = "testing", feature = "secure-srs"))]
#[derive(Clone, Copy, Debug)]
pub struct LedgerWithHeight<const H: u8>;

#[cfg(any(test, feature = "testing", feature = "secure-srs"))]
lazy_static! {
    static ref CAP_UNIVERSAL_PARAM: UniversalParam = jf_cap::proof::universal_setup_for_staging(
        2u64.pow(17) as usize,
        &mut ChaChaRng::from_seed([0u8; 32])
    )
    .unwrap();
}

#[cfg(any(test, feature = "testing", feature = "secure-srs"))]
impl<const H: u8> traits::Ledger for LedgerWithHeight<H> {
    type Validator = Validator<H>;

    fn name() -> String {
        String::from("Minimal CAP Ledger")
    }

    fn merkle_height() -> u8 {
        H
    }

    fn record_root_history() -> usize {
        1
    }

    fn srs() -> &'static UniversalParam {
        &*CAP_UNIVERSAL_PARAM
    }
}

/// The default ledger height is 5, for testing.
#[cfg(any(test, feature = "testing"))]
pub const DEFAULT_MERKLE_HEIGHT: u8 = 5;

#[cfg(any(test, feature = "testing"))]
pub type Ledger = LedgerWithHeight<DEFAULT_MERKLE_HEIGHT>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Ledger as _, NullifierSet as _, Transaction as _, Validator as _};
    use jf_cap::{
        freeze::{FreezeNote, FreezeNoteInput},
        keys::{FreezerKeyPair, UserKeyPair},
        structs::{AssetPolicy, FeeInput, FreezeFlag, RecordCommitment, RecordOpening, TxnFeeInfo},
        transfer::TransferNoteInput,
        AccMemberWitness, MerkleTree,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaChaRng};

    #[test]
    fn test_nullifier_set() {
        let mut rng = ChaChaRng::from_seed([42u8; 32]);

        let mut s = NullifierSet::default();
        assert_eq!(s.len(), 0);

        let n = Nullifier::random_for_test(&mut rng);
        s.multi_insert(&[(n, ())]).unwrap();
        assert_eq!(s.len(), 1);
        assert!(s.contains(&n.into()));
    }

    #[test]
    fn test_transaction() {
        let mut rng = ChaChaRng::from_seed([42u8; 32]);
        let key = UserKeyPair::generate(&mut rng);
        let freezer_key = FreezerKeyPair::generate(&mut rng);
        let viewer_key = ViewerKeyPair::generate(&mut rng);
        let srs = Ledger::srs();

        let xfr_proving_key =
            jf_cap::proof::transfer::preprocess(srs, 2, 2, Ledger::merkle_height())
                .unwrap()
                .0;
        let mint_proving_key = jf_cap::proof::mint::preprocess(srs, Ledger::merkle_height())
            .unwrap()
            .0;
        let freeze_proving_key = jf_cap::proof::freeze::preprocess(srs, 2, Ledger::merkle_height())
            .unwrap()
            .0;

        // Set up a ledger. For simplicity we will use the same ledger state and fee input for each
        // transaction (mint, transfer, and freeze);
        let mut records = MerkleTree::new(Ledger::merkle_height()).unwrap();
        let fee_ro = RecordOpening::new(
            &mut rng,
            1u8.into(),
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

        // To freeze and view, we need a record of a non-native asset type.
        let (asset_code, seed) = AssetCode::random(&mut rng);
        let policy = AssetPolicy::default()
            .set_freezer_pub_key(freezer_key.pub_key())
            .set_viewer_pub_key(viewer_key.pub_key())
            .reveal_user_address()
            .unwrap()
            .reveal_amount()
            .unwrap();
        let asset_def = AssetDefinition::new(asset_code, policy).unwrap();
        let asset_ro = RecordOpening::new(
            &mut rng,
            1u8.into(),
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
            ro: fee_ro,
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 0)
                .expect_ok()
                .unwrap()
                .1,
            owner_keypair: &key,
        };

        // Generate one transaction of each type.
        let mint_ro = RecordOpening::new(
            &mut rng,
            1u8.into(),
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let mint_comm = RecordCommitment::from(&mint_ro);
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1u8.into())
            .unwrap()
            .0;
        let mint_note =
            MintNote::generate(&mut rng, mint_ro, seed, &[], fee_info, &mint_proving_key)
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
            1u8.into(),
            asset_def.clone(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let xfr_comm = RecordCommitment::from(&xfr_ro);
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1u8.into())
            .unwrap()
            .0;
        let xfr_note = TransferNote::generate_non_native(
            &mut rng,
            xfr_inputs,
            &[xfr_ro],
            fee_info,
            2u64.pow(jf_cap::constants::MAX_TIMESTAMP_LEN as u32) - 1,
            &xfr_proving_key,
            vec![],
        )
        .unwrap()
        .0;
        let xfr = TransactionNote::Transfer(Box::new(xfr_note.clone()));

        // To freeze, we need a record of a non-native asset type.
        let freeze_inputs = vec![FreezeNoteInput {
            ro: asset_ro,
            acc_member_witness: AccMemberWitness::lookup_from_tree(&records, 1)
                .expect_ok()
                .unwrap()
                .1,
            keypair: &freezer_key,
        }];
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1u8.into())
            .unwrap()
            .0;
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

        // Check viewing interface.
        let viewable_assets = vec![(asset_code, asset_def.clone())].into_iter().collect();
        let viewing_keys = vec![(viewer_key.pub_key(), viewer_key)]
            .into_iter()
            .collect();

        let mint_memo = mint
            .open_viewing_memo(&viewable_assets, &viewing_keys)
            .unwrap();
        assert_eq!(mint_memo.asset, asset_def);
        assert_eq!(mint_memo.inputs, vec![]);
        assert_eq!(mint_memo.outputs.len(), 1);
        assert_eq!(mint_memo.outputs[0].asset_code, asset_code);
        assert_eq!(mint_memo.outputs[0].user_address, Some(key.address()));
        assert_eq!(mint_memo.outputs[0].amount, Some(1u8.into()));
        let xfr_memo = xfr
            .open_viewing_memo(&viewable_assets, &viewing_keys)
            .unwrap();
        assert_eq!(xfr_memo.asset, asset_def);
        assert_eq!(xfr_memo.inputs.len(), 1);
        assert_eq!(xfr_memo.inputs[0].asset_code, asset_code);
        assert_eq!(xfr_memo.inputs[0].user_address, Some(key.address()));
        assert_eq!(xfr_memo.inputs[0].amount, Some(1u8.into()));
        assert_eq!(xfr_memo.outputs.len(), 1);
        assert_eq!(xfr_memo.outputs[0].asset_code, asset_code);
        assert_eq!(xfr_memo.outputs[0].user_address, Some(key.address()));
        assert_eq!(xfr_memo.outputs[0].amount, Some(1u8.into()));

        assert_eq!(
            freeze.open_viewing_memo(&viewable_assets, &viewing_keys),
            Err(ViewingError::NoViewingMemos)
        );
    }

    #[test]
    fn test_validator() {
        let mut validator = Validator::<DEFAULT_MERKLE_HEIGHT>::default();
        assert_eq!(validator.now(), 0);
        let initial_commit = validator.commit();

        // Build a block to apply. For this we need a transaction note. It doesn't have to be valid, but
        // the easiest way to build a transaction note requires it to be valid, so we'll set up a whole
        // mock ledger.
        let mut rng = ChaChaRng::from_seed([42u8; 32]);
        let key = UserKeyPair::generate(&mut rng);
        let srs = Ledger::srs();
        let mint_proving_key = jf_cap::proof::mint::preprocess(srs, Ledger::merkle_height())
            .unwrap()
            .0;
        let mut records = MerkleTree::new(Ledger::merkle_height()).unwrap();
        let fee_ro = RecordOpening::new(
            &mut rng,
            1u8.into(),
            AssetDefinition::native(),
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let fee_comm = RecordCommitment::from(&fee_ro);
        records.push(fee_comm.to_field_element());
        let fee_input = FeeInput {
            ro: fee_ro,
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
            1u8.into(),
            asset_def,
            key.pub_key(),
            FreezeFlag::Unfrozen,
        );
        let fee_info = TxnFeeInfo::new(&mut rng, fee_input.clone(), 1u8.into())
            .unwrap()
            .0;
        let mint_note =
            MintNote::generate(&mut rng, mint_ro, seed, &[], fee_info, &mint_proving_key)
                .unwrap()
                .0;
        let mint = TransactionNote::Mint(Box::new(mint_note.clone()));

        // Apply a block and check that the correct UIDs and Merkle paths are computed.
        let (uids, records) = validator.validate_and_apply(vec![mint.clone()]).unwrap();
        assert_eq!(uids, vec![0, 1]);
        assert_eq!(records.num_leaves(), 2);
        assert_eq!(
            records.get_leaf(0).expect_ok().unwrap().1.leaf.0,
            mint_note.chg_comm.to_field_element()
        );
        assert_eq!(
            records.get_leaf(1).expect_ok().unwrap().1.leaf.0,
            mint_note.mint_comm.to_field_element()
        );

        // Make sure we have a new timestamp and commit.
        assert_eq!(validator.now(), 1);
        assert_ne!(validator.commit(), initial_commit);

        // Apply another block and check that we get different UIDs. Technically it's not allowed to
        // apply the same block twice, but our minimal validator doesn't care.
        let (uids, records) = validator.validate_and_apply(vec![mint]).unwrap();
        assert_eq!(uids, vec![2, 3]);
        assert_eq!(records.num_leaves(), 4);
        assert_eq!(
            records.get_leaf(2).expect_ok().unwrap().1.leaf.0,
            mint_note.chg_comm.to_field_element()
        );
        assert_eq!(
            records.get_leaf(3).expect_ok().unwrap().1.leaf.0,
            mint_note.mint_comm.to_field_element()
        );
    }
}
