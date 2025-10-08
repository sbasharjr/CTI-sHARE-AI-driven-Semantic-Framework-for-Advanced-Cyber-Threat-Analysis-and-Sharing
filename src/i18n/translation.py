"""
Multi-language support (Internationalization - i18n)
"""

import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TranslationManager:
    """
    Manager for multi-language translation support
    """
    
    def __init__(self, default_language: str = 'en'):
        """
        Initialize translation manager
        
        Args:
            default_language: Default language code (e.g., 'en', 'es', 'fr')
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ar', 'ru']
        
        self._load_default_translations()
        
    def _load_default_translations(self):
        """Load default translations for supported languages"""
        
        # English translations
        self.translations['en'] = {
            'threat_detected': 'Threat Detected',
            'severity': 'Severity',
            'category': 'Category',
            'confidence': 'Confidence',
            'timestamp': 'Timestamp',
            'description': 'Description',
            'critical': 'Critical',
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
            'informational': 'Informational',
            'malware': 'Malware',
            'phishing': 'Phishing',
            'ddos': 'DDoS Attack',
            'data_breach': 'Data Breach',
            'apt': 'Advanced Persistent Threat',
            'vulnerability_exploit': 'Vulnerability Exploit',
            'insider_threat': 'Insider Threat',
            'supply_chain': 'Supply Chain Attack',
            'alert_sent': 'Alert sent to security team',
            'ip_blocked': 'IP address blocked',
            'domain_blocked': 'Domain blocked',
            'file_quarantined': 'File quarantined',
            'host_isolated': 'Host isolated'
        }
        
        # Spanish translations
        self.translations['es'] = {
            'threat_detected': 'Amenaza Detectada',
            'severity': 'Severidad',
            'category': 'Categoría',
            'confidence': 'Confianza',
            'timestamp': 'Marca de Tiempo',
            'description': 'Descripción',
            'critical': 'Crítico',
            'high': 'Alto',
            'medium': 'Medio',
            'low': 'Bajo',
            'informational': 'Informativo',
            'malware': 'Malware',
            'phishing': 'Phishing',
            'ddos': 'Ataque DDoS',
            'data_breach': 'Filtración de Datos',
            'apt': 'Amenaza Persistente Avanzada',
            'vulnerability_exploit': 'Explotación de Vulnerabilidad',
            'insider_threat': 'Amenaza Interna',
            'supply_chain': 'Ataque a la Cadena de Suministro',
            'alert_sent': 'Alerta enviada al equipo de seguridad',
            'ip_blocked': 'Dirección IP bloqueada',
            'domain_blocked': 'Dominio bloqueado',
            'file_quarantined': 'Archivo en cuarentena',
            'host_isolated': 'Host aislado'
        }
        
        # French translations
        self.translations['fr'] = {
            'threat_detected': 'Menace Détectée',
            'severity': 'Gravité',
            'category': 'Catégorie',
            'confidence': 'Confiance',
            'timestamp': 'Horodatage',
            'description': 'Description',
            'critical': 'Critique',
            'high': 'Élevé',
            'medium': 'Moyen',
            'low': 'Faible',
            'informational': 'Informatif',
            'malware': 'Logiciel Malveillant',
            'phishing': 'Hameçonnage',
            'ddos': 'Attaque DDoS',
            'data_breach': 'Violation de Données',
            'apt': 'Menace Persistante Avancée',
            'vulnerability_exploit': 'Exploitation de Vulnérabilité',
            'insider_threat': 'Menace Interne',
            'supply_chain': 'Attaque de la Chaîne d\'Approvisionnement',
            'alert_sent': 'Alerte envoyée à l\'équipe de sécurité',
            'ip_blocked': 'Adresse IP bloquée',
            'domain_blocked': 'Domaine bloqué',
            'file_quarantined': 'Fichier mis en quarantaine',
            'host_isolated': 'Hôte isolé'
        }
        
        # German translations
        self.translations['de'] = {
            'threat_detected': 'Bedrohung Erkannt',
            'severity': 'Schweregrad',
            'category': 'Kategorie',
            'confidence': 'Vertrauen',
            'timestamp': 'Zeitstempel',
            'description': 'Beschreibung',
            'critical': 'Kritisch',
            'high': 'Hoch',
            'medium': 'Mittel',
            'low': 'Niedrig',
            'informational': 'Informativ',
            'malware': 'Schadsoftware',
            'phishing': 'Phishing',
            'ddos': 'DDoS-Angriff',
            'data_breach': 'Datenschutzverletzung',
            'apt': 'Fortgeschrittene Persistente Bedrohung',
            'vulnerability_exploit': 'Schwachstellen-Exploit',
            'insider_threat': 'Insider-Bedrohung',
            'supply_chain': 'Lieferkettenangriff',
            'alert_sent': 'Alarm an das Sicherheitsteam gesendet',
            'ip_blocked': 'IP-Adresse blockiert',
            'domain_blocked': 'Domain blockiert',
            'file_quarantined': 'Datei unter Quarantäne gestellt',
            'host_isolated': 'Host isoliert'
        }
        
        # Chinese translations
        self.translations['zh'] = {
            'threat_detected': '检测到威胁',
            'severity': '严重程度',
            'category': '类别',
            'confidence': '置信度',
            'timestamp': '时间戳',
            'description': '描述',
            'critical': '严重',
            'high': '高',
            'medium': '中',
            'low': '低',
            'informational': '信息',
            'malware': '恶意软件',
            'phishing': '钓鱼攻击',
            'ddos': 'DDoS攻击',
            'data_breach': '数据泄露',
            'apt': '高级持续性威胁',
            'vulnerability_exploit': '漏洞利用',
            'insider_threat': '内部威胁',
            'supply_chain': '供应链攻击',
            'alert_sent': '警报已发送给安全团队',
            'ip_blocked': 'IP地址已被封锁',
            'domain_blocked': '域名已被封锁',
            'file_quarantined': '文件已被隔离',
            'host_isolated': '主机已被隔离'
        }
        
        logger.info(f"Loaded translations for {len(self.translations)} languages")
    
    def set_language(self, language_code: str) -> bool:
        """
        Set current language
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language_code in self.translations:
            self.current_language = language_code
            logger.info(f"Language set to: {language_code}")
            return True
        else:
            logger.warning(f"Language not supported: {language_code}")
            return False
    
    def get_translation(self, key: str, language: Optional[str] = None) -> str:
        """
        Get translation for a key
        
        Args:
            key: Translation key
            language: Optional language code (uses current if not provided)
            
        Returns:
            Translated string
        """
        lang = language or self.current_language
        
        if lang not in self.translations:
            lang = self.default_language
        
        return self.translations[lang].get(key, key)
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Get translation with optional formatting
        
        Args:
            key: Translation key
            **kwargs: Format parameters
            
        Returns:
            Translated and formatted string
        """
        translation = self.get_translation(key)
        
        if kwargs:
            try:
                return translation.format(**kwargs)
            except KeyError:
                return translation
        
        return translation
    
    def translate_threat_data(self, threat_data: Dict[str, Any],
                             language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate threat data to specified language
        
        Args:
            threat_data: Threat information
            language: Target language (uses current if not provided)
            
        Returns:
            Translated threat data
        """
        lang = language or self.current_language
        
        translated = threat_data.copy()
        
        # Translate severity
        if 'severity' in translated:
            severity_key = translated['severity'].lower()
            translated['severity_translated'] = self.get_translation(severity_key, lang)
        
        # Translate category
        if 'category' in translated:
            category_key = translated['category'].lower().replace(' ', '_')
            translated['category_translated'] = self.get_translation(category_key, lang)
        
        return translated
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        
        Returns:
            List of language codes
        """
        return self.supported_languages
    
    def add_custom_translation(self, language: str, key: str, value: str) -> None:
        """
        Add custom translation
        
        Args:
            language: Language code
            key: Translation key
            value: Translation value
        """
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = value
        logger.info(f"Added custom translation for '{key}' in language '{language}'")
    
    def load_translations_from_file(self, filepath: str, language: str) -> bool:
        """
        Load translations from JSON file
        
        Args:
            filepath: Path to translation file
            language: Language code
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            if language not in self.translations:
                self.translations[language] = {}
            
            self.translations[language].update(translations)
            logger.info(f"Loaded translations from {filepath} for language '{language}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load translations from {filepath}: {e}")
            return False
    
    def save_translations_to_file(self, filepath: str, language: str) -> bool:
        """
        Save translations to JSON file
        
        Args:
            filepath: Path to save file
            language: Language code
            
        Returns:
            True if saved successfully
        """
        try:
            if language not in self.translations:
                logger.warning(f"No translations found for language '{language}'")
                return False
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved translations to {filepath} for language '{language}'")
            return True
        except Exception as e:
            logger.error(f"Failed to save translations to {filepath}: {e}")
            return False


# Global translation manager instance
_translation_manager = None


def get_translation_manager() -> TranslationManager:
    """
    Get global translation manager instance
    
    Returns:
        TranslationManager instance
    """
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def translate(key: str, language: Optional[str] = None) -> str:
    """
    Convenience function to get translation
    
    Args:
        key: Translation key
        language: Optional language code
        
    Returns:
        Translated string
    """
    return get_translation_manager().get_translation(key, language)
