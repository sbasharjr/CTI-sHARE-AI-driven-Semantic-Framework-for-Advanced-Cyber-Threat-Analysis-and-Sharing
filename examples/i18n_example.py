"""
Example: Multi-language Support (i18n)
"""

from src.i18n.translation import TranslationManager, get_translation_manager


def example_translation():
    """Example of multi-language support"""
    print("=" * 80)
    print("Multi-language Support (Internationalization)")
    print("=" * 80)
    
    # Get translation manager
    print("\n1. Initializing translation manager...")
    tm = get_translation_manager()
    
    print(f"   Default language: {tm.default_language}")
    print(f"   Supported languages: {', '.join(tm.get_supported_languages())}")
    
    # Demonstrate translations in different languages
    print("\n2. Translating common threat terms...")
    print("=" * 80)
    
    terms = ['threat_detected', 'severity', 'critical', 'malware', 'phishing']
    languages = ['en', 'es', 'fr', 'de', 'zh']
    
    # Print header
    print(f"\n{'Term':<20}", end='')
    for lang in languages:
        print(f"{lang.upper():<15}", end='')
    print()
    print("-" * 90)
    
    # Print translations
    for term in terms:
        print(f"{term:<20}", end='')
        for lang in languages:
            translation = tm.get_translation(term, language=lang)
            print(f"{translation:<15}", end='')
        print()
    
    # Translate threat data
    print("\n3. Translating threat data to different languages...")
    print("=" * 80)
    
    threat_data = {
        'id': 'T001',
        'severity': 'CRITICAL',
        'category': 'malware',
        'description': 'Ransomware attack detected on network',
        'confidence': 0.95
    }
    
    print(f"\nOriginal threat data:")
    print(f"   ID: {threat_data['id']}")
    print(f"   Severity: {threat_data['severity']}")
    print(f"   Category: {threat_data['category']}")
    print(f"   Description: {threat_data['description']}")
    
    # Translate to different languages
    for lang_code in ['es', 'fr', 'de']:
        translated = tm.translate_threat_data(threat_data, language=lang_code)
        
        lang_names = {
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese'
        }
        
        print(f"\n{lang_names[lang_code]} ({lang_code.upper()}):")
        print(f"   ID: {translated['id']}")
        print(f"   {tm.get_translation('severity', lang_code)}: {translated['severity_translated']}")
        print(f"   {tm.get_translation('category', lang_code)}: {translated['category_translated']}")
    
    # Demonstrate action translations
    print("\n4. Translating response actions...")
    print("=" * 80)
    
    actions = ['ip_blocked', 'domain_blocked', 'file_quarantined', 'alert_sent']
    
    for action in actions:
        print(f"\n{action}:")
        for lang in ['en', 'es', 'fr', 'de']:
            translation = tm.get_translation(action, language=lang)
            lang_names = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German'}
            print(f"   {lang_names[lang]:<10}: {translation}")
    
    # Change current language
    print("\n5. Setting system language...")
    print("-" * 80)
    
    for lang in ['es', 'fr', 'de', 'en']:
        tm.set_language(lang)
        threat_msg = tm.get_translation('threat_detected')
        severity_msg = tm.get_translation('severity')
        critical_msg = tm.get_translation('critical')
        
        lang_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German'
        }
        
        print(f"\n{lang_names[lang]} ({lang.upper()}):")
        print(f"   {threat_msg}: {severity_msg} = {critical_msg}")
    
    # Add custom translation
    print("\n6. Adding custom translations...")
    tm.add_custom_translation('en', 'zero_day', 'Zero-Day Vulnerability')
    tm.add_custom_translation('es', 'zero_day', 'Vulnerabilidad de Día Cero')
    tm.add_custom_translation('fr', 'zero_day', 'Vulnérabilité Zero-Day')
    
    print("   Added custom term 'zero_day' in multiple languages:")
    for lang in ['en', 'es', 'fr']:
        translation = tm.get_translation('zero_day', language=lang)
        print(f"      {lang.upper()}: {translation}")
    
    # Demonstrate usage in alert messages
    print("\n7. Example: Multi-language threat alerts...")
    print("=" * 80)
    
    alert_template = {
        'en': "{threat}: {severity} - {category}",
        'es': "{threat}: {severity} - {category}",
        'fr': "{threat}: {severity} - {category}",
        'de': "{threat}: {severity} - {category}"
    }
    
    for lang in ['en', 'es', 'fr', 'de']:
        threat = tm.get_translation('threat_detected', lang)
        severity = tm.get_translation('critical', lang)
        category = tm.get_translation('malware', lang)
        
        alert = alert_template[lang].format(
            threat=threat,
            severity=severity,
            category=category
        )
        
        lang_names = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German'}
        print(f"\n{lang_names[lang]}:")
        print(f"   {alert}")
    
    print("\n" + "=" * 80)
    print("Benefits of Multi-language Support:")
    print("=" * 80)
    print("✓ Global accessibility: Teams worldwide can use the system")
    print("✓ Better communication: Alerts in native language")
    print("✓ Compliance: Meet regional language requirements")
    print("✓ User experience: Improved understanding and response time")
    print("=" * 80)


if __name__ == "__main__":
    try:
        example_translation()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
