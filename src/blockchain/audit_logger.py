"""
Blockchain audit trail implementation using Hyperledger Fabric
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class BlockchainAuditLogger:
    """Blockchain audit logger for credit decisions"""
    
    def __init__(self):
        self.network_config = {
            'peer_url': 'peer0.org1.example.com:7051',
            'orderer_url': 'orderer.example.com:7050',
            'channel_name': 'finriskai-channel',
            'chaincode_name': 'credit-audit'
        }
        self.private_key, self.public_key = self._generate_keypair()
        
    def _generate_keypair(self):
        """Generate RSA keypair for signing"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def _sign_data(self, data: str) -> str:
        """Sign data with private key"""
        signature = self.private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()
    
    def _create_audit_record(self, application_id: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit record for blockchain"""
        timestamp = datetime.utcnow().isoformat()
        
        # Core audit data
        audit_data = {
            'application_id': application_id,
            'timestamp': timestamp,
            'decision': decision_data.get('decision'),
            'credit_score': decision_data.get('credit_score'),
            'risk_grade': decision_data.get('risk_grade'),
            'confidence_score': decision_data.get('confidence_score'),
            'model_version': '1.0.0',
            'features_used': list(decision_data.get('features', {}).keys()),
            'explanation_hash': self._hash_explanation(decision_data.get('explanation', {}))
        }
        
        # Create data hash
        data_string = json.dumps(audit_data, sort_keys=True)
        data_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        # Sign the hash
        signature = self._sign_data(data_hash)
        
        # Complete audit record
        audit_record = {
            **audit_data,
            'data_hash': data_hash,
            'signature': signature,
            'public_key': self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
        }
        
        return audit_record
    
    def _hash_explanation(self, explanation: Dict[str, Any]) -> str:
        """Create hash of explanation data"""
        explanation_string = json.dumps(explanation, sort_keys=True)
        return hashlib.sha256(explanation_string.encode()).hexdigest()
    
    def log_credit_decision(self, application_id: str, decision_data: Dict[str, Any]) -> str:
        """Log credit decision to blockchain"""
        try:
            # Create audit record
            audit_record = self._create_audit_record(application_id, decision_data)
            
            # Submit to blockchain (simplified implementation)
            transaction_id = self._submit_to_blockchain(audit_record)
            
            logger.info(f"Audit record logged to blockchain: {transaction_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error logging to blockchain: {e}")
            raise
    
    def _submit_to_blockchain(self, audit_record: Dict[str, Any]) -> str:
        """Submit audit record to blockchain network"""
        # This is a simplified implementation
        # In production, use actual Hyperledger Fabric SDK
        
        # Simulate blockchain transaction
        import uuid
        transaction_id = str(uuid.uuid4())
        
        # Store locally for demo (in production, this goes to blockchain)
        audit_file = f"blockchain_audit_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            with open(audit_file, 'a') as f:
                audit_entry = {
                    'transaction_id': transaction_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'audit_record': audit_record
                }
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing audit file: {e}")
        
        return transaction_id
    
    def verify_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Verify a blockchain transaction"""
        # Implementation would query blockchain for transaction
        # For demo, we'll read from local file
        
        audit_files = [f for f in os.listdir('.') if f.startswith('blockchain_audit_')]
        
        for audit_file in audit_files:
            try:
                with open(audit_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry['transaction_id'] == transaction_id:
                            return {
                                'found': True,
                                'transaction_id': transaction_id,
                                'audit_record': entry['audit_record']
                            }
            except Exception as e:
                logger.error(f"Error reading audit file {audit_file}: {e}")
        
        return {'found': False, 'transaction_id': transaction_id}
    
    def get_audit_trail(self, application_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for an application"""
        audit_trail = []
        audit_files = [f for f in os.listdir('.') if f.startswith('blockchain_audit_')]
        
        for audit_file in audit_files:
            try:
                with open(audit_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry['audit_record']['application_id'] == application_id:
                            audit_trail.append(entry)
            except Exception as e:
                logger.error(f"Error reading audit file {audit_file}: {e}")
        
        # Sort by timestamp
        audit_trail.sort(key=lambda x: x['audit_record']['timestamp'])
        return audit_trail