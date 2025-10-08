"""
Blockchain-based Threat Intelligence Verification System
"""

import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Block:
    """
    Represents a block in the blockchain
    """
    
    def __init__(self, index: int, timestamp: str, data: Dict[str, Any],
                 previous_hash: str, nonce: int = 0):
        """
        Initialize block
        
        Args:
            index: Block index in chain
            timestamp: Block creation timestamp
            data: Block data (threat intelligence)
            previous_hash: Hash of previous block
            nonce: Nonce for proof of work
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate hash of the block
        
        Returns:
            Block hash
        """
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        Mine block with proof of work
        
        Args:
            difficulty: Mining difficulty (number of leading zeros)
        """
        target = '0' * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        logger.info(f"Block mined: {self.hash}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary
        
        Returns:
            Block as dictionary
        """
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }


class ThreatIntelligenceBlockchain:
    """
    Blockchain for threat intelligence verification and immutable storage
    """
    
    def __init__(self, difficulty: int = 2):
        """
        Initialize blockchain
        
        Args:
            difficulty: Mining difficulty for proof of work
        """
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_threats: List[Dict[str, Any]] = []
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self) -> None:
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            data={'genesis': True, 'message': 'Threat Intelligence Blockchain Genesis Block'},
            previous_hash='0'
        )
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def get_latest_block(self) -> Block:
        """
        Get the latest block in the chain
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def add_threat(self, threat_data: Dict[str, Any]) -> None:
        """
        Add threat intelligence to pending threats
        
        Args:
            threat_data: Threat information
        """
        # Add metadata
        threat_entry = {
            'threat_id': threat_data.get('id', 'unknown'),
            'timestamp': threat_data.get('timestamp', datetime.now().isoformat()),
            'severity': threat_data.get('severity', 'MEDIUM'),
            'category': threat_data.get('category', 'unknown'),
            'entities': threat_data.get('entities', {}),
            'confidence': threat_data.get('confidence', 0.0),
            'source': threat_data.get('source', 'unknown'),
            'hash': self._hash_threat_data(threat_data)
        }
        
        self.pending_threats.append(threat_entry)
        logger.info(f"Added threat to pending queue: {threat_entry['threat_id']}")
    
    def _hash_threat_data(self, threat_data: Dict[str, Any]) -> str:
        """
        Create hash of threat data for verification
        
        Args:
            threat_data: Threat information
            
        Returns:
            Hash of threat data
        """
        threat_string = json.dumps(threat_data, sort_keys=True)
        return hashlib.sha256(threat_string.encode()).hexdigest()
    
    def mine_pending_threats(self) -> Optional[Block]:
        """
        Mine pending threats into a new block
        
        Returns:
            Newly created block or None if no pending threats
        """
        if not self.pending_threats:
            logger.warning("No pending threats to mine")
            return None
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data={
                'threats': self.pending_threats,
                'count': len(self.pending_threats)
            },
            previous_hash=self.get_latest_block().hash
        )
        
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add to chain
        self.chain.append(new_block)
        
        # Clear pending threats
        self.pending_threats = []
        
        logger.info(f"New block mined and added to chain: Block #{new_block.index}")
        return new_block
    
    def verify_chain(self) -> bool:
        """
        Verify integrity of the blockchain
        
        Returns:
            True if chain is valid
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block's hash is correct
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Invalid hash at block {i}")
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash at block {i}")
                return False
            
            # Check proof of work
            if not current_block.hash.startswith('0' * self.difficulty):
                logger.error(f"Invalid proof of work at block {i}")
                return False
        
        logger.info("Blockchain is valid")
        return True
    
    def get_threat_by_id(self, threat_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve threat from blockchain by ID
        
        Args:
            threat_id: Threat identifier
            
        Returns:
            Threat data if found
        """
        for block in self.chain:
            if 'threats' in block.data:
                for threat in block.data['threats']:
                    if threat.get('threat_id') == threat_id:
                        return {
                            'threat': threat,
                            'block_index': block.index,
                            'block_hash': block.hash,
                            'verified': True
                        }
        return None
    
    def verify_threat(self, threat_id: str, threat_hash: str) -> bool:
        """
        Verify if a threat exists in blockchain with matching hash
        
        Args:
            threat_id: Threat identifier
            threat_hash: Expected threat hash
            
        Returns:
            True if threat is verified
        """
        threat_info = self.get_threat_by_id(threat_id)
        
        if threat_info:
            stored_hash = threat_info['threat'].get('hash')
            if stored_hash == threat_hash:
                logger.info(f"Threat verified: {threat_id}")
                return True
            else:
                logger.warning(f"Threat hash mismatch: {threat_id}")
                return False
        
        logger.warning(f"Threat not found in blockchain: {threat_id}")
        return False
    
    def get_threats_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get all threats with specified severity
        
        Args:
            severity: Threat severity level
            
        Returns:
            List of matching threats
        """
        threats = []
        for block in self.chain:
            if 'threats' in block.data:
                for threat in block.data['threats']:
                    if threat.get('severity') == severity:
                        threats.append(threat)
        return threats
    
    def get_threats_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all threats with specified category
        
        Args:
            category: Threat category
            
        Returns:
            List of matching threats
        """
        threats = []
        for block in self.chain:
            if 'threats' in block.data:
                for threat in block.data['threats']:
                    if threat.get('category') == category:
                        threats.append(threat)
        return threats
    
    def get_chain_statistics(self) -> Dict[str, Any]:
        """
        Get blockchain statistics
        
        Returns:
            Statistics dictionary
        """
        total_blocks = len(self.chain)
        total_threats = sum(
            block.data.get('count', 0)
            for block in self.chain
            if 'threats' in block.data
        )
        
        return {
            'total_blocks': total_blocks,
            'total_threats': total_threats,
            'chain_length': total_blocks,
            'pending_threats': len(self.pending_threats),
            'is_valid': self.verify_chain(),
            'latest_block_hash': self.get_latest_block().hash
        }
    
    def export_chain(self) -> List[Dict[str, Any]]:
        """
        Export entire blockchain
        
        Returns:
            List of blocks as dictionaries
        """
        return [block.to_dict() for block in self.chain]
    
    def import_chain(self, chain_data: List[Dict[str, Any]]) -> bool:
        """
        Import blockchain from data
        
        Args:
            chain_data: List of blocks as dictionaries
            
        Returns:
            True if import successful
        """
        try:
            new_chain = []
            
            for block_data in chain_data:
                block = Block(
                    index=block_data['index'],
                    timestamp=block_data['timestamp'],
                    data=block_data['data'],
                    previous_hash=block_data['previous_hash'],
                    nonce=block_data['nonce']
                )
                block.hash = block_data['hash']
                new_chain.append(block)
            
            # Verify imported chain
            self.chain = new_chain
            if self.verify_chain():
                logger.info("Blockchain imported successfully")
                return True
            else:
                logger.error("Imported blockchain is invalid")
                return False
        except Exception as e:
            logger.error(f"Failed to import blockchain: {e}")
            return False


class BlockchainVerificationService:
    """
    Service for verifying and managing threat intelligence on blockchain
    """
    
    def __init__(self, difficulty: int = 2):
        """
        Initialize blockchain verification service
        
        Args:
            difficulty: Mining difficulty
        """
        self.blockchain = ThreatIntelligenceBlockchain(difficulty)
        
    def submit_threat(self, threat_data: Dict[str, Any]) -> str:
        """
        Submit threat to blockchain
        
        Args:
            threat_data: Threat information
            
        Returns:
            Threat hash for verification
        """
        self.blockchain.add_threat(threat_data)
        return self.blockchain._hash_threat_data(threat_data)
    
    def commit_threats(self) -> Optional[Block]:
        """
        Commit pending threats to blockchain
        
        Returns:
            New block or None
        """
        return self.blockchain.mine_pending_threats()
    
    def verify_threat_authenticity(self, threat_id: str, threat_hash: str) -> bool:
        """
        Verify threat authenticity
        
        Args:
            threat_id: Threat identifier
            threat_hash: Expected threat hash
            
        Returns:
            True if authentic
        """
        return self.blockchain.verify_threat(threat_id, threat_hash)
    
    def get_verified_threat(self, threat_id: str) -> Optional[Dict[str, Any]]:
        """
        Get verified threat from blockchain
        
        Args:
            threat_id: Threat identifier
            
        Returns:
            Threat information if found
        """
        return self.blockchain.get_threat_by_id(threat_id)
