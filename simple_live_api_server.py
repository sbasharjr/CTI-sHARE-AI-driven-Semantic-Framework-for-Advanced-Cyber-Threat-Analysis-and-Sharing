#!/usr/bin/env python3
"""
Simple Flask server to test Live Data API endpoints
"""

from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import random
import logging

app = Flask(__name__)
CORS(app)

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock threat history for testing
threat_history = [
    {
        'timestamp': datetime.now().isoformat(),
        'content': 'malware detected in network traffic with suspicious behavior',
        'category': 'malware'
    },
    {
        'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
        'content': 'phishing email containing fake login credentials detected',
        'category': 'phishing'
    },
    {
        'timestamp': (datetime.now() - timedelta(minutes=45)).isoformat(),
        'content': 'brute force attack on ssh server from multiple ips',
        'category': 'brute_force'
    }
]

@app.route('/api/dashboard/live/system-performance')
def get_live_system_performance():
    """Get live system performance data for timeline chart"""
    try:
        try:
            import psutil
            
            # Get real system performance metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': round(cpu_percent, 1),
                'memory_usage': round(memory.percent, 1),
                'disk_usage': round((disk.used / disk.total) * 100, 1),
                'memory_available': round((memory.available / memory.total) * 100, 1),
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'processes_count': len(psutil.pids())
            })
        except ImportError:
            # Fallback to simulated data if psutil not available
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': round(random.uniform(10, 80), 1),
                'memory_usage': round(random.uniform(30, 85), 1), 
                'disk_usage': round(random.uniform(20, 70), 1),
                'memory_available': round(random.uniform(15, 70), 1),
                'network_bytes_sent': random.randint(1000000, 10000000),
                'network_bytes_recv': random.randint(1000000, 10000000),
                'load_average': round(random.uniform(0.5, 4.0), 2),
                'processes_count': random.randint(150, 400)
            })
    except Exception as e:
        logger.error(f"Error fetching system performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/resource-distribution')
def get_live_resource_distribution():
    """Get live resource distribution data"""
    try:
        try:
            import psutil
            
            # Get real memory information
            memory = psutil.virtual_memory()
            
            # Calculate distribution percentages
            available_percent = round((memory.available / memory.total) * 100, 1)
            used_percent = round(memory.percent, 1)
            
            # Estimate system vs application usage
            system_percent = round(used_percent * 0.3, 1)  # ~30% for system
            apps_percent = round(used_percent * 0.7, 1)    # ~70% for applications
            cached_percent = round((memory.cached / memory.total) * 100, 1) if hasattr(memory, 'cached') else round(used_percent * 0.15, 1)
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'available': available_percent,
                'system': system_percent,
                'applications': apps_percent,
                'cached': cached_percent,
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'used_memory_gb': round(memory.used / (1024**3), 2),
                'free_memory_gb': round(memory.free / (1024**3), 2)
            })
        except ImportError:
            # Fallback to simulated data
            available = round(random.uniform(25, 60), 1)
            system = round(random.uniform(10, 25), 1)
            apps = round(random.uniform(20, 45), 1)
            cached = round(100 - available - system - apps, 1)
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'available': available,
                'system': system,
                'applications': apps,
                'cached': cached,
                'total_memory_gb': 16.0,
                'used_memory_gb': round((100 - available) * 0.16, 2),
                'free_memory_gb': round(available * 0.16, 2)
            })
    except Exception as e:
        logger.error(f"Error fetching resource distribution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/attack-vectors')
def get_live_attack_vectors():
    """Get live attack vectors data from threat history"""
    try:
        # Analyze recent threats from history
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_threats = [t for t in threat_history 
                        if 'timestamp' in t and datetime.fromisoformat(t['timestamp']) > recent_threshold]
        
        # Count attack vector types
        attack_vectors = {
            'malware': 0,
            'phishing': 0,
            'brute_force': 0,
            'sql_injection': 0,
            'ddos': 0,
            'privilege_escalation': 0,
            'social_engineering': 0,
            'ransomware': 0
        }
        
        # Analyze threats for attack patterns
        for threat in recent_threats:
            content = str(threat.get('content', '')).lower()
            
            if any(word in content for word in ['malware', 'virus', 'trojan', 'worm']):
                attack_vectors['malware'] += 1
            elif any(word in content for word in ['phishing', 'phish', 'fake', 'spoofing']):
                attack_vectors['phishing'] += 1
            elif any(word in content for word in ['brute', 'force', 'password', 'login']):
                attack_vectors['brute_force'] += 1
            elif any(word in content for word in ['sql', 'injection', 'database']):
                attack_vectors['sql_injection'] += 1
            elif any(word in content for word in ['ddos', 'dos', 'flood', 'attack']):
                attack_vectors['ddos'] += 1
        
        # Add baseline realistic numbers
        base_multiplier = max(1, len(recent_threats))
        
        for vector in attack_vectors:
            if attack_vectors[vector] == 0:
                attack_vectors[vector] = random.randint(50, 300) * base_multiplier
            else:
                attack_vectors[vector] += random.randint(10, 100)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'vectors': attack_vectors,
            'total_attacks': sum(attack_vectors.values()),
            'analysis_period': '1 hour',
            'threat_count': len(recent_threats)
        })
    except Exception as e:
        logger.error(f"Error fetching attack vectors: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/geographic-distribution')
def get_live_geographic_distribution():
    """Get live geographic threat distribution"""
    try:
        # Geographic regions and their threat counts
        regions = {
            'United States': random.randint(800, 1500),
            'China': random.randint(600, 1200), 
            'Russia': random.randint(500, 1000),
            'Germany': random.randint(400, 800),
            'United Kingdom': random.randint(300, 700),
            'Brazil': random.randint(250, 600),
            'India': random.randint(200, 500),
            'Japan': random.randint(150, 400),
            'South Korea': random.randint(100, 300),
            'Canada': random.randint(80, 250)
        }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'regions': regions,
            'total_threats': sum(regions.values()),
            'analysis_period': '2 hours',
            'top_region': max(regions.items(), key=lambda x: x[1])
        })
    except Exception as e:
        logger.error(f"Error fetching geographic distribution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/hourly-activity')
def get_live_hourly_activity():
    """Get live hourly activity pattern data"""
    try:
        # Generate realistic hourly pattern
        now = datetime.now()
        current_hour = now.hour
        hourly_data = {}
        
        for hour in range(24):
            # Business hours pattern (8 AM to 6 PM higher activity)
            if 8 <= hour <= 18:
                base_activity = random.randint(300, 800)
            elif 19 <= hour <= 23 or 6 <= hour <= 7:
                base_activity = random.randint(150, 400)  # Evening/morning
            else:
                base_activity = random.randint(50, 200)   # Night
            
            # Current hour spike
            if hour == current_hour:
                base_activity += random.randint(200, 500)
            
            hourly_data[hour] = base_activity
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'hourly_data': hourly_data,
            'current_hour': current_hour,
            'total_today': sum(hourly_data.values()),
            'peak_hour': max(hourly_data.items(), key=lambda x: x[1])[0],
            'analysis_period': '24 hours'
        })
    except Exception as e:
        logger.error(f"Error fetching hourly activity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/feed-status')
def get_live_feed_status():
    """Get live threat intelligence feed status timeline"""
    try:
        feeds = [
            'MISP Community', 'AlienVault OTX', 'VirusTotal',
            'Emerging Threats', 'ThreatFox', 'Abuse.ch',
            'URLVoid', 'IBM X-Force'
        ]
        
        # Generate timeline data for last 15 minutes
        timeline_data = []
        
        for i in range(15):
            timestamp = datetime.now() - timedelta(minutes=14-i)
            health_score = random.randint(85, 99)
            active_feeds = random.randint(6, len(feeds))
            
            timeline_data.append({
                'timestamp': timestamp.isoformat(),
                'health_score': health_score,
                'active_feeds': active_feeds,
                'total_feeds': len(feeds)
            })
        
        # Current feed statuses
        feed_statuses = {}
        for feed in feeds:
            feed_statuses[feed] = {
                'status': 'active' if random.random() > 0.1 else 'warning',
                'health': random.randint(80, 100),
                'last_update': (datetime.now() - timedelta(minutes=random.randint(1, 15))).isoformat(),
                'ioc_count': random.randint(500, 5000),
                'response_time': random.randint(50, 500)
            }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'timeline': timeline_data,
            'feed_statuses': feed_statuses,
            'overall_health': random.randint(88, 97),
            'total_feeds': len(feeds),
            'active_feeds': sum(1 for f in feed_statuses.values() if f['status'] == 'active'),
            'total_iocs': sum(f['ioc_count'] for f in feed_statuses.values())
        })
    except Exception as e:
        logger.error(f"Error fetching feed status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/live/ioc-types')
def get_live_ioc_types():
    """Get live IOC types distribution"""
    try:
        ioc_types = {
            'ip_addresses': random.randint(1000, 5000),
            'domains': random.randint(800, 4000),
            'urls': random.randint(600, 3000),
            'file_hashes': random.randint(500, 2500),
            'email_addresses': random.randint(300, 1500),
            'cve_ids': random.randint(200, 1000),
            'registry_keys': random.randint(150, 800),
            'file_paths': random.randint(100, 600)
        }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'ioc_types': ioc_types,
            'total_iocs': sum(ioc_types.values()),
            'analysis_period': '1 hour',
            'most_common': max(ioc_types.items(), key=lambda x: x[1]),
            'threat_sources': len(threat_history)
        })
    except Exception as e:
        logger.error(f"Error fetching IOC types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Simple index page"""
    return jsonify({
        'message': 'Live Data API Server Running',
        'endpoints': [
            '/api/dashboard/live/system-performance',
            '/api/dashboard/live/resource-distribution', 
            '/api/dashboard/live/attack-vectors',
            '/api/dashboard/live/geographic-distribution',
            '/api/dashboard/live/hourly-activity',
            '/api/dashboard/live/feed-status',
            '/api/dashboard/live/ioc-types'
        ]
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🔴 Live Data API Server Starting...")
    print("📡 Available endpoints:")
    print("   - /api/dashboard/live/system-performance")
    print("   - /api/dashboard/live/resource-distribution") 
    print("   - /api/dashboard/live/attack-vectors")
    print("   - /api/dashboard/live/geographic-distribution")
    print("   - /api/dashboard/live/hourly-activity")
    print("   - /api/dashboard/live/feed-status")
    print("   - /api/dashboard/live/ioc-types")
    print("🌐 Server running at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)