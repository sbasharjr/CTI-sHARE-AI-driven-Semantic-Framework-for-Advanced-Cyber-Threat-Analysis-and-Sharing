"""
Automated Threat Response System
"""

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import logging
import threading
import time

logger = logging.getLogger(__name__)


class ResponseAction:
    """
    Base class for automated response actions
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize response action
        
        Args:
            name: Action name
            description: Action description
        """
        self.name = name
        self.description = description
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the response action
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        raise NotImplementedError("Subclasses must implement execute")


class BlockIPAction(ResponseAction):
    """
    Action to block IP addresses
    """
    
    def __init__(self):
        super().__init__("block_ip", "Block malicious IP addresses")
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Block IP addresses identified in threat
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        ips = threat_data.get('entities', {}).get('ips', [])
        
        blocked_ips = []
        for ip in ips:
            # In a real implementation, this would interact with firewall/IDS
            logger.info(f"Blocking IP: {ip}")
            blocked_ips.append(ip)
        
        return {
            'action': self.name,
            'success': True,
            'blocked_ips': blocked_ips,
            'timestamp': datetime.now().isoformat()
        }


class BlockDomainAction(ResponseAction):
    """
    Action to block domains
    """
    
    def __init__(self):
        super().__init__("block_domain", "Block malicious domains")
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Block domains identified in threat
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        domains = threat_data.get('entities', {}).get('domains', [])
        
        blocked_domains = []
        for domain in domains:
            # In a real implementation, this would interact with DNS/proxy
            logger.info(f"Blocking domain: {domain}")
            blocked_domains.append(domain)
        
        return {
            'action': self.name,
            'success': True,
            'blocked_domains': blocked_domains,
            'timestamp': datetime.now().isoformat()
        }


class QuarantineFileAction(ResponseAction):
    """
    Action to quarantine files
    """
    
    def __init__(self):
        super().__init__("quarantine_file", "Quarantine suspicious files")
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quarantine files identified in threat
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        hashes = threat_data.get('entities', {}).get('hashes', [])
        
        quarantined = []
        for file_hash in hashes:
            # In a real implementation, this would interact with endpoint protection
            logger.info(f"Quarantining file with hash: {file_hash}")
            quarantined.append(file_hash)
        
        return {
            'action': self.name,
            'success': True,
            'quarantined_files': quarantined,
            'timestamp': datetime.now().isoformat()
        }


class IsolateHostAction(ResponseAction):
    """
    Action to isolate compromised hosts
    """
    
    def __init__(self):
        super().__init__("isolate_host", "Isolate compromised hosts from network")
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Isolate hosts involved in threat
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        hosts = threat_data.get('affected_hosts', [])
        
        isolated = []
        for host in hosts:
            # In a real implementation, this would interact with network control
            logger.info(f"Isolating host: {host}")
            isolated.append(host)
        
        return {
            'action': self.name,
            'success': True,
            'isolated_hosts': isolated,
            'timestamp': datetime.now().isoformat()
        }


class AlertSecurityTeamAction(ResponseAction):
    """
    Action to alert security team
    """
    
    def __init__(self, email_addresses: List[str] = None, slack_webhook: str = None):
        super().__init__("alert_team", "Alert security team")
        self.email_addresses = email_addresses or []
        self.slack_webhook = slack_webhook
        
    def execute(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send alert to security team
        
        Args:
            threat_data: Threat information
            
        Returns:
            Action result
        """
        alert_message = self._format_alert(threat_data)
        
        # In a real implementation, this would send emails/slack messages
        logger.info(f"Alerting security team: {alert_message}")
        
        return {
            'action': self.name,
            'success': True,
            'alert_sent': True,
            'recipients': len(self.email_addresses),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_alert(self, threat_data: Dict[str, Any]) -> str:
        """Format alert message"""
        return (f"THREAT ALERT: {threat_data.get('text', 'Unknown threat')} | "
                f"Severity: {threat_data.get('severity', 'MEDIUM')} | "
                f"Confidence: {threat_data.get('confidence', 0):.2%}")


class ResponseRule:
    """
    Rule defining when and how to respond to threats
    """
    
    def __init__(self, name: str, conditions: Dict[str, Any], actions: List[ResponseAction]):
        """
        Initialize response rule
        
        Args:
            name: Rule name
            conditions: Conditions that trigger this rule
            actions: Actions to execute when rule triggers
        """
        self.name = name
        self.conditions = conditions
        self.actions = actions
        
    def matches(self, threat_data: Dict[str, Any]) -> bool:
        """
        Check if threat matches rule conditions
        
        Args:
            threat_data: Threat information
            
        Returns:
            True if conditions are met
        """
        # Check severity condition
        if 'min_severity' in self.conditions:
            severity_levels = {'INFORMATIONAL': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5}
            threat_severity = severity_levels.get(threat_data.get('severity', 'MEDIUM'), 3)
            min_severity = severity_levels.get(self.conditions['min_severity'], 3)
            
            if threat_severity < min_severity:
                return False
        
        # Check confidence condition
        if 'min_confidence' in self.conditions:
            if threat_data.get('confidence', 0) < self.conditions['min_confidence']:
                return False
        
        # Check category condition
        if 'categories' in self.conditions:
            threat_category = threat_data.get('category', '')
            if threat_category not in self.conditions['categories']:
                return False
        
        # Check if entities present
        if 'requires_entities' in self.conditions:
            required = self.conditions['requires_entities']
            entities = threat_data.get('entities', {})
            for entity_type in required:
                if not entities.get(entity_type):
                    return False
        
        return True
    
    def execute(self, threat_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute all actions in this rule
        
        Args:
            threat_data: Threat information
            
        Returns:
            List of action results
        """
        results = []
        
        for action in self.actions:
            try:
                result = action.execute(threat_data)
                results.append(result)
                logger.info(f"Rule '{self.name}': Executed action '{action.name}'")
            except Exception as e:
                logger.error(f"Rule '{self.name}': Failed to execute action '{action.name}': {e}")
                results.append({
                    'action': action.name,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results


class AutomatedResponseSystem:
    """
    Main automated threat response system
    """
    
    def __init__(self):
        """Initialize automated response system"""
        self.rules: List[ResponseRule] = []
        self.response_history: List[Dict[str, Any]] = []
        self.is_running = False
        self._lock = threading.Lock()
        
    def add_rule(self, rule: ResponseRule) -> None:
        """
        Add response rule
        
        Args:
            rule: Response rule to add
        """
        with self._lock:
            self.rules.append(rule)
        logger.info(f"Added response rule: {rule.name}")
        
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove response rule by name
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        with self._lock:
            for i, rule in enumerate(self.rules):
                if rule.name == rule_name:
                    self.rules.pop(i)
                    logger.info(f"Removed response rule: {rule_name}")
                    return True
        return False
    
    def process_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process threat and execute appropriate responses
        
        Args:
            threat_data: Threat information
            
        Returns:
            Response summary
        """
        response_summary = {
            'threat_id': threat_data.get('id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'rules_matched': [],
            'actions_executed': [],
            'success': True
        }
        
        # Check each rule
        for rule in self.rules:
            if rule.matches(threat_data):
                response_summary['rules_matched'].append(rule.name)
                
                # Execute rule actions
                action_results = rule.execute(threat_data)
                response_summary['actions_executed'].extend(action_results)
                
                # Check if any action failed
                for result in action_results:
                    if not result.get('success', False):
                        response_summary['success'] = False
        
        # Store in history
        with self._lock:
            self.response_history.append(response_summary)
            
            # Keep only last 1000 entries
            if len(self.response_history) > 1000:
                self.response_history = self.response_history[-1000:]
        
        logger.info(f"Processed threat: {len(response_summary['rules_matched'])} rules matched, "
                   f"{len(response_summary['actions_executed'])} actions executed")
        
        return response_summary
    
    def get_response_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get response history
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of response summaries
        """
        with self._lock:
            return self.response_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get response statistics
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_responses = len(self.response_history)
            successful_responses = sum(1 for r in self.response_history if r.get('success', False))
            
            action_counts = {}
            for response in self.response_history:
                for action in response.get('actions_executed', []):
                    action_name = action.get('action', 'unknown')
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            return {
                'total_responses': total_responses,
                'successful_responses': successful_responses,
                'success_rate': successful_responses / total_responses if total_responses > 0 else 0,
                'active_rules': len(self.rules),
                'action_counts': action_counts
            }
    
    def create_default_rules(self) -> None:
        """Create default response rules"""
        
        # Rule for critical threats
        critical_rule = ResponseRule(
            name="critical_threat_response",
            conditions={'min_severity': 'CRITICAL', 'min_confidence': 0.8},
            actions=[
                BlockIPAction(),
                BlockDomainAction(),
                IsolateHostAction(),
                AlertSecurityTeamAction()
            ]
        )
        self.add_rule(critical_rule)
        
        # Rule for high severity threats
        high_severity_rule = ResponseRule(
            name="high_severity_response",
            conditions={'min_severity': 'HIGH', 'min_confidence': 0.7},
            actions=[
                BlockIPAction(),
                BlockDomainAction(),
                AlertSecurityTeamAction()
            ]
        )
        self.add_rule(high_severity_rule)
        
        # Rule for malware threats
        malware_rule = ResponseRule(
            name="malware_response",
            conditions={'categories': ['malware'], 'min_confidence': 0.6},
            actions=[
                QuarantineFileAction(),
                AlertSecurityTeamAction()
            ]
        )
        self.add_rule(malware_rule)
        
        logger.info("Created default response rules")
