//! Minimal implementation of the ledger traits for an AAP-style ledger. This implementation can be
//! used as an aid in implemeneting the traits for more complex, specific ledger types. It is also
//! fully functional and can be used as a mock ledger for testing downstream modules that are
//! parameterized by ledger type.

use crate::traits;
use crate::types::{AuditError, AuditMemoOpening};
use jf_aap::{
    keys::{AuditorKeyPair, AuditorPubKey},
    mint::MintNote,
    structs::{AssetCode, AssetDefinition},
    transfer::TransferNote,
    TransactionNote,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

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

pub fn open_audit_memo(
    assets: &HashMap<AssetCode, AssetDefinition>,
    keys: &HashMap<AuditorPubKey, AuditorKeyPair>,
    txn: &TransactionNote,
) -> Result<AuditMemoOpening, AuditError> {
    match txn {
        TransactionNote::Transfer(xfr) => open_xfr_audit_memo(assets, keys, xfr),
        TransactionNote::Mint(mint) => open_mint_audit_memo(keys, mint),
        TransactionNote::Freeze(_) => Err(AuditError::NoAuditMemos),
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
