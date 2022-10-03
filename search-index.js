var searchIndex = JSON.parse('{\
"reef":{"doc":"An abstraction of a CAP-style network.","t":[2,0,0,0,3,13,13,3,13,6,13,13,6,4,13,13,4,3,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,5,5,11,11,12,12,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,8,16,16,16,16,8,8,16,16,16,16,8,16,8,8,8,16,10,10,10,10,10,10,11,10,11,10,11,10,10,10,10,10,10,10,10,11,11,10,10,10,10,10,10,10,10,10,10,6,6,13,6,6,6,6,6,6,13,6,6,4,3,11,11,11,11,12,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,11,11,11,11,11,11,11,11,11,12,11,11,11,11,11,11,11,11,11,11],"n":["Ledger","cap","traits","types","Block","Failed","Freeze","LedgerWithHeight","Mint","NullifierSet","Receive","Send","Transaction","TransactionKind","Unfreeze","Unknown","ValidationError","Validator","add_transaction","arbitrary","as_any","as_any","as_any","as_any","as_any","as_any_mut","as_any_mut","as_any_mut","as_any_mut","as_any_mut","as_error_source","backtrace","block_height","block_height","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","cause","clone","clone","clone","clone","clone","clone_into","clone_into","clone_into","clone_into","clone_into","commit","default","deref","deref","deref","deref","deref","deref_mut","deref_mut","deref_mut","deref_mut","deref_mut","description","deserialize","deserialize","deserialize","deserialize","drop","drop","drop","drop","drop","eq","eq","eq","fmt","fmt","fmt","fmt","fmt","fmt","fmt","freeze","from","from","from","from","from","get_hash","get_hash","hash","hash","index","index","init","init","init","init","init","into","into","into","into","into","into_any","into_any","into_any","into_any","into_any","into_any_arc","into_any_arc","into_any_arc","into_any_arc","into_any_arc","into_any_rc","into_any_rc","into_any_rc","into_any_rc","into_any_rc","into_iter","into_iter","is_bad_nullifier_proof","is_empty","iter","len","merkle_height","mint","multi_insert","name","ne","ne","new","next_block","open_mint_viewing_memo","open_xfr_viewing_memo","receive","record_root_history","records_commitment","records_frontier","send","serialize","serialize","serialize","serialize","source","srs","to_owned","to_owned","to_owned","to_owned","to_owned","to_string","to_string","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","txns","type_id","type_id","type_id","type_id","type_id","unfreeze","unknown","validate_and_apply","vzip","vzip","vzip","vzip","vzip","msg","Block","Block","Error","Hash","Kind","Ledger","NullifierSet","NullifierSet","Proof","Proof","StateCommitment","Transaction","Transaction","TransactionKind","ValidationError","Validator","Validator","add_transaction","block_height","cap","commit","freeze","hash","input_nullifiers","is_bad_nullifier_proof","is_empty","kind","len","merkle_height","mint","multi_insert","name","new","next_block","open_viewing_memo","output_commitments","output_len","output_openings","proven_nullifiers","receive","record_root_history","send","set_proofs","srs","txns","unfreeze","unknown","validate_and_apply","Block","BlockProof","NoViewingMemos","NullifierProof","NullifierSet","StateCommitment","Transaction","TransactionHash","TransactionKind","UnviewableAsset","ValidationError","Validator","ViewingError","ViewingMemoOpening","as_any","as_any","as_any_mut","as_any_mut","asset","borrow","borrow","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","deref","deref","deref_mut","deref_mut","drop","drop","eq","eq","fmt","fmt","from","from","init","init","inputs","into","into","into_any","into_any","into_any_arc","into_any_arc","into_any_rc","into_any_rc","ne","outputs","to_owned","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","vzip","vzip"],"q":["reef","","","","reef::cap","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","reef::cap::ValidationError","reef::traits","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","reef::types","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["","Minimal implementation of the ledger traits for a …","Traits describing the interface of a CAP ledger.","Ledger types","A block of CAP transactions.","","","A minimal CAP ledger.","","A set of nullifiers.","","","A CAP transaction.","All the kinds of transactions in the basic CAP protocol.","","","Errors in mock CAP validation.","A mock CAP validator.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","","","","","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Attempt to open the viewer memo attached to a CAP mint …","Attempt to open the viewer memo attached to a CAP transfer …","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","A block of transactions.","Blocks applied by this validator.","Errors that can occur when validation this block.","Transaction digest.","Supported transaction types.","A CAP ledger.","A set of nullifiers which have been spent.","Nullifier set to be updated when a transaction is added to …","Authentication that a given nullifier is or isn’t in the …","Additional data required to authenticate a committed …","A commitment to the state of the validator.","A CAP transaction.","Transactions in this block.","The types of transactions supported by this network.","Errors that can occur when validating a Transaction or …","State required to validate a Block of Transactions.","The state of a validator for this ledger.","Add a new Transaction to this block.","The number of blocks this validator has applied.","Wrap a raw CAP transaction in this network’s transaction …","The commitment to the current state of the validator.","","A committing hash of this transaction.","This transaction’s inputs, without authentication.","Whether validation failed because a transaction’s …","Whether there are no transactions in this block.","The type of this transaction.","The number of transactions in this block.","The height of the ledger’s records MerkleTree.","","Update the set to include additional nullifiers.","A human-readable name for this ledger.","A catch-all error variant with a helpful message.","Build a block on top of the current state of this …","Attempt to decrypt the attached viewing memo.","Commitments to the records that this transaction will …","The number of outputs.","If this is not a private transaction, get the openings of …","Nullifiers for the records that this transaction will …","","The number of past MerkleTree roots maintained by …","","Update the proofs that this transaction’s nullifiers are …","The universal setup for PLONK proofs.","The transactions in this block.","","","Apply a new block, updating the state and returning UIDs …","A block of transactions that can be applied to a ledger <code>L</code>.","A proof used to authenticate a committed block in a ledger …","","A proof that a nullifier is spent or unspent, relative to …","A set of spent nullifiers for a ledger <code>L</code>.","A commitment to a validator state for a ledger <code>L</code>.","A transaction that can be applied to a ledger <code>L</code>.","A committing hash of a transaction that can be applied to …","Types of transactions supported by a ledger <code>L</code>.","","An error that can occur while validating transitions of a …","A validator for a ledger <code>L</code>.","Errors that can occur when trying to decrypt a viewing …","Information contained in a viewing memo.","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","",""],"i":[0,0,0,0,0,9,13,0,13,0,13,13,0,0,13,13,0,0,1,5,13,9,1,5,14,13,9,1,5,14,9,9,5,5,13,9,1,5,14,13,9,1,5,14,9,13,9,1,5,14,13,9,1,5,14,5,5,13,9,1,5,14,13,9,1,5,14,9,13,9,1,5,13,9,1,5,14,13,1,5,13,13,9,9,1,5,14,13,13,9,1,5,14,13,1,13,1,1,1,13,9,1,5,14,13,9,1,5,14,13,9,1,5,14,13,9,1,5,14,13,9,1,5,14,1,1,9,1,1,1,14,13,27,14,1,5,9,5,0,0,13,14,5,5,13,13,9,1,5,9,14,13,9,1,5,14,13,9,13,9,1,5,14,13,9,1,5,14,1,13,9,1,5,14,13,13,5,13,9,1,5,14,42,0,43,44,45,45,0,0,45,46,43,43,0,44,0,0,0,47,44,43,45,43,48,45,45,49,44,45,44,47,48,46,47,49,43,45,45,45,45,45,48,47,48,45,47,44,48,48,43,0,0,33,0,0,0,0,0,0,33,0,0,0,0,32,33,32,33,32,32,33,32,33,32,33,32,33,32,33,32,33,32,33,32,33,32,33,32,33,32,33,32,32,33,32,33,32,33,32,33,32,32,32,33,32,33,32,33,32,33,32,33],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[1,2],3],[4,[[6,[5]]]],[[],7],[[],7],[[],7],[[],7],[[],7],[[],7],[[],7],[[],7],[[],7],[[],7],[[],8],[9,[[11,[10]]]],[5,12],0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[9,[[11,[8]]]],[13,13],[9,9],[1,1],[5,5],[14,14],[[]],[[]],[[]],[[]],[[]],[5],[[],5],[15],[15],[15],[15],[15],[15],[15],[15],[15],[15],[9,16],[[],[[3,[13]]]],[[],[[3,[9]]]],[[],[[3,[1]]]],[[],[[3,[5]]]],[15],[15],[15],[15],[15],[[13,13],17],[[1,1],17],[[5,5],17],[[13,18],[[3,[19]]]],[[13,18],20],[[9,18],20],[[9,18],20],[[1,18],20],[[5,18],20],[[14,18],20],[[],13],[[]],[[]],[[]],[[]],[[]],[[],12],[[],12],[13],[1],[1,12],[[1,15],2],[[],15],[[],15],[[],15],[[],15],[[],15],[[]],[[]],[[]],[[]],[[]],[[[22,[21]]],[[22,[7,21]]]],[[[22,[21]]],[[22,[7,21]]]],[[[22,[21]]],[[22,[7,21]]]],[[[22,[21]]],[[22,[7,21]]]],[[[22,[21]]],[[22,[7,21]]]],[23,[[23,[7]]]],[23,[[23,[7]]]],[23,[[23,[7]]]],[23,[[23,[7]]]],[23,[[23,[7]]]],[24,[[24,[7]]]],[24,[[24,[7]]]],[24,[[24,[7]]]],[24,[[24,[7]]]],[24,[[24,[7]]]],[1],[1],[9,17],[1,17],[1,25],[1,15],[[],26],[[],13],[27,3],[[],28],[[1,1],17],[[5,5],17],[29,9],[5],[[30,31],[[3,[32,33]]]],[[30,30,34],[[3,[32,33]]]],[[],13],[[],15],0,0,[[],13],[13,3],[9,3],[1,3],[5,3],[9,[[11,[8]]]],[[],35],[[]],[[]],[[]],[[]],[[]],[[],28],[[],28],[[],3],[[],3],[[],3],[[],3],[[],3],[[],3],[[],3],[[],3],[[],3],[[],3],[1,[[36,[2]]]],[[],37],[[],37],[[],37],[[],37],[[],37],[[],13],[[],13],[5,[[3,[9]]]],[[]],[[]],[[]],[[]],[[]],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[],3],[[],12],[[38,36]],[[]],[[]],[[]],[[],[[36,[39]]]],[[],17],[[],17],[[]],[[],15],[[],26],[[]],[[],3],[[],28],[29],[[]],[[30,30],[[3,[32,33]]]],[[],[[36,[40]]]],[[],15],[[],[[11,[[36,[41]]]]]],[[],36],[[]],[[],15],[[]],[36],[[],35],[[],36],[[]],[[]],[[],3],0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[],7],[[],7],[[],7],[[],7],0,[[]],[[]],[[]],[[]],[32,32],[33,33],[[]],[[]],[15],[15],[15],[15],[15],[15],[[32,32],17],[[33,33],17],[[32,18],20],[[33,18],20],[[]],[[]],[[],15],[[],15],0,[[]],[[]],[[[22,[21]]],[[22,[7,21]]]],[[[22,[21]]],[[22,[7,21]]]],[23,[[23,[7]]]],[23,[[23,[7]]]],[24,[[24,[7]]]],[24,[[24,[7]]]],[[32,32],17],0,[[]],[[]],[[],3],[[],3],[[],3],[[],3],[[],37],[[],37],[[]],[[]]],"p":[[3,"Block"],[6,"Transaction"],[4,"Result"],[3,"Unstructured"],[3,"Validator"],[6,"Result"],[8,"Any"],[8,"Error"],[4,"ValidationError"],[3,"Backtrace"],[4,"Option"],[15,"u64"],[4,"TransactionKind"],[3,"LedgerWithHeight"],[15,"usize"],[15,"str"],[15,"bool"],[3,"Formatter"],[3,"Error"],[6,"Result"],[3,"Global"],[3,"Box"],[3,"Arc"],[3,"Rc"],[8,"Iterator"],[15,"u8"],[6,"NullifierSet"],[3,"String"],[8,"Display"],[3,"HashMap"],[3,"MintNote"],[3,"ViewingMemoOpening"],[4,"ViewingError"],[3,"TransferNote"],[6,"UniversalParam"],[3,"Vec"],[3,"TypeId"],[4,"TransactionNote"],[3,"Nullifier"],[3,"RecordCommitment"],[3,"RecordOpening"],[13,"Failed"],[8,"Validator"],[8,"Block"],[8,"Transaction"],[8,"NullifierSet"],[8,"Ledger"],[8,"TransactionKind"],[8,"ValidationError"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
