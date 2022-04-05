var searchIndex = JSON.parse('{\
"reef":{"doc":"An abstraction of a CAP-style network.","t":[11,0,11,11,0,11,0,6,13,13,3,13,6,13,13,6,4,13,13,4,3,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,5,5,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,8,16,16,16,16,8,8,16,16,16,8,16,8,8,8,16,10,10,10,10,10,11,10,11,10,11,10,10,10,10,10,10,10,10,10,11,11,10,10,10,10,10,10,10,10,10,10,4,3,6,13,6,6,6,6,6,6,13,6,6,11,11,11,11,12,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,11,11,11,11,11,11,11,11,11,12,11,11,11,11,11,11,11,11,11,11],"n":["add_transaction","cap","multi_insert","new","traits","txns","types","Block","Failed","Freeze","Ledger","Mint","NullifierSet","Receive","Send","Transaction","TransactionKind","Unfreeze","Unknown","ValidationError","Validator","arbitrary","arbitrary_take_rest","as_any","as_any","as_any","as_any","as_any_mut","as_any_mut","as_any_mut","as_any_mut","as_error_source","backtrace","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","cause","clone","clone","clone","clone","clone_into","clone_into","clone_into","clone_into","commit","default","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","description","deserialize","deserialize","deserialize","drop","drop","drop","drop","eq","eq","fmt","fmt","fmt","fmt","fmt","fmt","freeze","from","from","from","from","get_hash","hash","init","init","init","init","into","into","into","into","into_any","into_any","into_any","into_any","into_any_arc","into_any_arc","into_any_arc","into_any_arc","into_any_rc","into_any_rc","into_any_rc","into_any_rc","is_bad_nullifier_proof","merkle_height","mint","name","ne","new","now","now","num_records","open_mint_audit_memo","open_xfr_audit_memo","receive","record_root_history","send","serialize","serialize","serialize","size_hint","source","srs","to_owned","to_owned","to_owned","to_owned","to_string","to_string","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","unfreeze","unknown","validate_and_apply","vzip","vzip","vzip","vzip","msg","Block","Block","Error","Hash","Kind","Ledger","NullifierSet","NullifierSet","Proof","StateCommitment","Transaction","Transaction","TransactionKind","ValidationError","Validator","Validator","add_transaction","cap","commit","freeze","hash","input_nullifiers","is_bad_nullifier_proof","is_empty","kind","len","merkle_height","mint","multi_insert","name","new","new","now","open_audit_memo","output_commitments","output_len","output_openings","proven_nullifiers","receive","record_root_history","send","set_proofs","srs","txns","unfreeze","unknown","validate_and_apply","AuditError","AuditMemoOpening","Block","NoAuditMemos","NullifierProof","NullifierSet","StateCommitment","Transaction","TransactionHash","TransactionKind","UnauditableAsset","ValidationError","Validator","as_any","as_any","as_any_mut","as_any_mut","asset","borrow","borrow","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","deref","deref","deref_mut","deref_mut","drop","drop","eq","eq","fmt","fmt","from","from","init","init","inputs","into","into","into_any","into_any","into_any_arc","into_any_arc","into_any_rc","into_any_rc","ne","outputs","to_owned","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","vzip","vzip"],"q":["reef","","","","","","","reef::cap","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","reef::cap::ValidationError","reef::traits","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","reef::types","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["","Minimal implementation of the ledger traits for a …","","","Traits describing the interface of a CAP ledger.","","Ledger types","A block of CAP transactions.","","","A minimal CAP ledger.","","A set of nullifiers.","","","A CAP transaction.","All the kinds of transactions in the basic CAP protocol.","","","Errors in mock CAP validation.","A mock CAP validator.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Attempt to open the viewer memo attached to a CAP mint …","Attempt to open the viewer memo attached to a CAP transfer …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","A block of transactions.","Blocks applied by this validator.","Errors that can occur when validation this block.","Transaction digest.","Supported transaction types.","A CAP ledger.","A set of nullifiers which have been spent.","Nullifier set to be updated when a transaction is added to …","Authentication that a given nullifier is or isn’t in the …","A commitment to the state of the validator.","A CAP transaction.","Transactions in this block.","The types of transactions supported by this network.","Errors that can occur when validating a Transaction or …","State required to validate a Block of Transactions.","The state of a validator for this ledger.","Add a new Transaction to this block.","Wrap a raw CAP transaction in this network’s transaction …","The commitment to the current state of the validator.","","A committing hash of this transaction.","This transaction’s inputs, without authentication.","Whether validation failed because a transaction’s …","Whether there are no transactions in this block.","The type of this transaction.","The number of transactions in this block.","The height of the ledger’s records MerkleTree.","","Update the set to include additional nullifiers.","A human-readable name for this ledger.","A catch-all error variant with a helpful message.","Create a block from a list of transactions.","The number of blocks this validator has applied.","Attempt to decrypt the attached viewing memo.","Commitments to the records created by this transaction.","The number of outputs.","If this is not a private transaction, get the openings of …","This transaction’s input nullifiers.","","The number of past MerkleTree roots maintained by …","","Update the proofs that this transaction’s nullifiers are …","The universal setup for PLONK proofs.","The transactions in this block.","","","Apply a new block, updating the state and returning UIDs …","Errors that can occur when trying to decrypt a viewing …","Information contained in a viewing memo.","A block of transactions that can be applied to a ledger <code>L</code>.","","A proof that a nullifier is spent or unspent, relative to …","A set of spent nullifiers for a ledger <code>L</code>.","A commitment to a validator state for a ledger <code>L</code>.","A transaction that can be applied to a ledger <code>L</code>.","A committing hash of a transaction that can be applied to …","Types of transactions supported by a ledger <code>L</code>.","","An error that can occur while validating transitions of a …","A validator for a ledger <code>L</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"i":[1,0,2,1,0,1,0,0,3,4,0,4,0,4,4,0,0,4,4,0,0,5,5,4,3,5,6,4,3,5,6,3,3,4,3,5,6,4,3,5,6,3,4,3,5,6,4,3,5,6,5,5,4,3,5,6,4,3,5,6,3,4,3,5,4,3,5,6,4,5,4,4,3,3,5,6,4,4,3,5,6,4,4,4,3,5,6,4,3,5,6,4,3,5,6,4,3,5,6,4,3,5,6,3,6,4,6,5,3,5,5,5,0,0,4,6,4,4,3,5,5,3,6,4,3,5,6,4,3,4,3,5,6,4,3,5,6,4,3,5,6,4,4,5,4,3,5,6,7,0,8,9,10,10,0,0,10,11,8,0,9,0,0,0,12,9,10,8,13,10,10,14,9,10,9,12,13,11,12,14,9,8,10,10,10,10,10,13,12,13,10,12,9,13,13,8,0,0,0,15,0,0,0,0,0,0,15,0,0,16,15,16,15,16,16,15,16,15,16,15,16,15,16,15,16,15,16,15,16,15,16,15,16,15,16,15,16,16,15,16,15,16,15,16,15,16,16,16,15,16,15,16,15,16,15,16,15],"f":[[[["transaction",6]],["result",4]],null,[[],["result",4]],[[["vec",3,[["transaction",6]]]]],null,[[],["vec",3,[["transaction",6]]]],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[[["unstructured",3]],["result",6]],[[["unstructured",3]],["result",6]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["error",8]],[[],["option",4,[["backtrace",3]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],["option",4,[["error",8]]]],[[],["transactionkind",4]],[[],["validationerror",4]],[[],["validator",3]],[[],["ledger",3]],[[]],[[]],[[]],[[]],[[]],[[],["validator",3]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[],["str",15]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["transactionkind",4]],["bool",15]],[[["validator",3]],["bool",15]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",4,[["error",3]]]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[]],[[]],[[]],[[]],[[]],[[],["u64",15]],[[]],[[],["usize",15]],[[],["usize",15]],[[],["usize",15]],[[],["usize",15]],[[]],[[]],[[]],[[]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[],["bool",15]],[[],["u8",15]],[[]],[[],["string",3]],[[["validator",3]],["bool",15]],[[]],[[],["u64",15]],null,null,[[["hashmap",3],["mintnote",3]],["result",4,[["auditmemoopening",3],["auditerror",4]]]],[[["hashmap",3],["hashmap",3],["transfernote",3]],["result",4,[["auditmemoopening",3],["auditerror",4]]]],[[]],[[],["usize",15]],[[]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[["usize",15]]],[[],["option",4,[["error",8]]]],[[],["universalparam",6]],[[]],[[]],[[]],[[]],[[],["string",3]],[[],["string",3]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[],["typeid",3]],[[]],[[]],[[],["result",4,[["vec",3,[["u64",15]]],["validationerror",4]]]],[[]],[[]],[[]],[[]],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[[],["result",4]],[[["transactionnote",4],["vec",3]]],[[]],[[]],[[]],[[],["vec",3,[["nullifier",3]]]],[[],["bool",15]],[[],["bool",15]],[[]],[[],["usize",15]],[[],["u8",15]],[[]],[[],["result",4]],[[],["string",3]],[[]],[[["vec",3]]],[[],["u64",15]],[[["hashmap",3],["hashmap",3]],["result",4,[["auditmemoopening",3],["auditerror",4]]]],[[],["vec",3,[["recordcommitment",3]]]],[[],["usize",15]],[[],["option",4,[["vec",3,[["recordopening",3]]]]]],[[],["vec",3]],[[]],[[],["usize",15]],[[]],[[["vec",3]]],[[],["universalparam",6]],[[],["vec",3]],[[]],[[]],[[],["result",4,[["vec",3,[["u64",15]]]]]],null,null,null,null,null,null,null,null,null,null,null,null,null,[[],["any",8]],[[],["any",8]],[[],["any",8]],[[],["any",8]],null,[[]],[[]],[[]],[[]],[[],["auditmemoopening",3]],[[],["auditerror",4]],[[]],[[]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["usize",15]]],[[["auditmemoopening",3]],["bool",15]],[[["auditerror",4]],["bool",15]],[[["formatter",3]],["result",6]],[[["formatter",3]],["result",6]],[[]],[[]],[[],["usize",15]],[[],["usize",15]],null,[[]],[[]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["box",3,[["global",3]]]],["box",3,[["any",8],["global",3]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["arc",3]],["arc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[["rc",3]],["rc",3,[["any",8]]]],[[["auditmemoopening",3]],["bool",15]],null,[[]],[[]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["typeid",3]],[[],["typeid",3]],[[]],[[]]],"p":[[6,"Block"],[6,"NullifierSet"],[4,"ValidationError"],[4,"TransactionKind"],[3,"Validator"],[3,"Ledger"],[13,"Failed"],[8,"Validator"],[8,"Block"],[8,"Transaction"],[8,"NullifierSet"],[8,"Ledger"],[8,"TransactionKind"],[8,"ValidationError"],[4,"AuditError"],[3,"AuditMemoOpening"]]}\
}');
if (window.initSearch) {window.initSearch(searchIndex)};