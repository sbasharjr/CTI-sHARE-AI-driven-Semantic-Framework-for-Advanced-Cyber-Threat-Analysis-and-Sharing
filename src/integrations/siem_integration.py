"""
Integration module for SIEM (Security Information and Event Management) systems
"""

import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SIEMConnector:
    """
    Base class for SIEM system connectors
    """
    
    def __init__(self, host: str, port: int, api_key: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize SIEM connector
        
        Args:
            host: SIEM server host
            port: SIEM server port
            api_key: API key for authentication
            username: Username for authentication
            password: Password for authentication
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.username = username
        self.password = password
        self.base_url = f"https://{host}:{port}"
        
    def test_connection(self) -> bool:
        """Test connection to SIEM system"""
        raise NotImplementedError("Subclasses must implement test_connection")
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to SIEM system"""
        raise NotImplementedError("Subclasses must implement send_alert")
    
    def query_events(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Query events from SIEM system"""
        raise NotImplementedError("Subclasses must implement query_events")


class SplunkConnector(SIEMConnector):
    """
    Connector for Splunk SIEM
    """
    
    def __init__(self, host: str, port: int = 8089, api_key: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Splunk connector
        
        Args:
            host: Splunk server host
            port: Splunk management port (default 8089)
            api_key: API token
            username: Username for authentication
            password: Password for authentication
        """
        super().__init__(host, port, api_key, username, password)
        self.session = requests.Session()
        
        # Setup authentication
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
    def test_connection(self) -> bool:
        """
        Test connection to Splunk
        
        Returns:
            True if connection successful
        """
        try:
            url = f"{self.base_url}/services/server/info"
            
            if self.api_key:
                response = self.session.get(url, verify=False)
            else:
                response = self.session.get(url, auth=(self.username, self.password), verify=False)
            
            if response.status_code == 200:
                logger.info("Successfully connected to Splunk")
                return True
            else:
                logger.error(f"Failed to connect to Splunk: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Splunk: {e}")
            return False
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert to Splunk
        
        Args:
            alert: Alert data
            
        Returns:
            True if alert sent successfully
        """
        try:
            url = f"{self.base_url}/services/receivers/simple"
            
            # Format alert for Splunk
            event_data = {
                'time': alert.get('timestamp', datetime.now().isoformat()),
                'source': 'threat_detection_framework',
                'sourcetype': 'threat_alert',
                'event': json.dumps(alert)
            }
            
            if self.api_key:
                response = self.session.post(url, data=event_data, verify=False)
            else:
                response = self.session.post(url, data=event_data,
                                            auth=(self.username, self.password), verify=False)
            
            if response.status_code in [200, 201]:
                logger.info(f"Alert sent to Splunk: {alert.get('id', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to send alert to Splunk: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error sending alert to Splunk: {e}")
            return False
    
    def query_events(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Query events from Splunk
        
        Args:
            query: Splunk search query
            start_time: Start time for query
            end_time: End time for query
            
        Returns:
            List of events
        """
        try:
            url = f"{self.base_url}/services/search/jobs"
            
            search_query = f'search {query} earliest="{start_time.isoformat()}" latest="{end_time.isoformat()}"'
            
            if self.api_key:
                response = self.session.post(url, data={'search': search_query}, verify=False)
            else:
                response = self.session.post(url, data={'search': search_query},
                                            auth=(self.username, self.password), verify=False)
            
            if response.status_code == 201:
                # Extract job ID and wait for results
                # Simplified - real implementation would poll for results
                logger.info(f"Search job created in Splunk")
                return []
            else:
                logger.error(f"Failed to query Splunk: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error querying Splunk: {e}")
            return []


class QRadarConnector(SIEMConnector):
    """
    Connector for IBM QRadar SIEM
    """
    
    def __init__(self, host: str, port: int = 443, api_key: str = None):
        """
        Initialize QRadar connector
        
        Args:
            host: QRadar server host
            port: QRadar API port (default 443)
            api_key: QRadar API token
        """
        super().__init__(host, port, api_key)
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'SEC': api_key})
        
    def test_connection(self) -> bool:
        """
        Test connection to QRadar
        
        Returns:
            True if connection successful
        """
        try:
            url = f"{self.base_url}/api/system/about"
            response = self.session.get(url, verify=False)
            
            if response.status_code == 200:
                logger.info("Successfully connected to QRadar")
                return True
            else:
                logger.error(f"Failed to connect to QRadar: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to QRadar: {e}")
            return False
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Send alert to QRadar as custom event
        
        Args:
            alert: Alert data
            
        Returns:
            True if alert sent successfully
        """
        try:
            url = f"{self.base_url}/api/siem/offenses"
            
            # Format alert for QRadar
            offense_data = {
                'description': alert.get('description', 'Threat detected'),
                'severity': alert.get('severity', 5),
                'magnitude': alert.get('magnitude', 5),
                'source_address_ids': alert.get('source_ips', [])
            }
            
            response = self.session.post(url, json=offense_data, verify=False)
            
            if response.status_code in [200, 201]:
                logger.info(f"Alert sent to QRadar: {alert.get('id', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to send alert to QRadar: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error sending alert to QRadar: {e}")
            return False
    
    def query_events(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Query events from QRadar
        
        Args:
            query: AQL query string
            start_time: Start time for query
            end_time: End time for query
            
        Returns:
            List of events
        """
        try:
            url = f"{self.base_url}/api/ariel/searches"
            
            # Convert times to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            aql_query = f"{query} START {start_ms} STOP {end_ms}"
            
            response = self.session.post(url, json={'query_expression': aql_query}, verify=False)
            
            if response.status_code == 201:
                logger.info(f"Search initiated in QRadar")
                return []
            else:
                logger.error(f"Failed to query QRadar: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error querying QRadar: {e}")
            return []


class SIEMIntegration:
    """
    Main SIEM integration class supporting multiple SIEM systems
    """
    
    def __init__(self):
        """Initialize SIEM integration"""
        self.connectors: Dict[str, SIEMConnector] = {}
        
    def add_siem(self, name: str, connector: SIEMConnector) -> None:
        """
        Add SIEM connector
        
        Args:
            name: Name for this SIEM connection
            connector: SIEM connector instance
        """
        self.connectors[name] = connector
        logger.info(f"Added SIEM connector: {name}")
        
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test all SIEM connections
        
        Returns:
            Dictionary of connection test results
        """
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.test_connection()
        return results
    
    def send_alert_to_all(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """
        Send alert to all configured SIEM systems
        
        Args:
            alert: Alert data
            
        Returns:
            Dictionary of send results
        """
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.send_alert(alert)
        return results
    
    def send_alert(self, siem_name: str, alert: Dict[str, Any]) -> bool:
        """
        Send alert to specific SIEM
        
        Args:
            siem_name: Name of SIEM to send to
            alert: Alert data
            
        Returns:
            True if alert sent successfully
        """
        if siem_name not in self.connectors:
            logger.error(f"SIEM connector not found: {siem_name}")
            return False
        
        return self.connectors[siem_name].send_alert(alert)
    
    def format_threat_alert(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format threat data as SIEM alert
        
        Args:
            threat_data: Threat detection data
            
        Returns:
            Formatted alert
        """
        alert = {
            'id': threat_data.get('id', 'unknown'),
            'timestamp': threat_data.get('timestamp', datetime.now().isoformat()),
            'title': threat_data.get('text', 'Threat detected')[:100],
            'description': threat_data.get('text', 'Unknown threat'),
            'severity': self._map_severity(threat_data.get('severity', 'MEDIUM')),
            'magnitude': threat_data.get('confidence', 5),
            'category': threat_data.get('category', 'unknown'),
            'source_ips': threat_data.get('entities', {}).get('ips', []),
            'domains': threat_data.get('entities', {}).get('domains', []),
            'confidence': threat_data.get('confidence', 0.0),
            'is_threat': threat_data.get('is_threat', False)
        }
        
        return alert
    
    def _map_severity(self, severity: str) -> int:
        """
        Map severity string to numeric value
        
        Args:
            severity: Severity string (CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL)
            
        Returns:
            Numeric severity (1-10)
        """
        severity_map = {
            'CRITICAL': 10,
            'HIGH': 8,
            'MEDIUM': 5,
            'LOW': 3,
            'INFORMATIONAL': 1
        }
        return severity_map.get(severity.upper(), 5)
