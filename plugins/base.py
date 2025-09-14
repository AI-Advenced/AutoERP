# autoerp/plugins/base.py
class BasePlugin:
    name = None
    version = None
    
    def install(self, erp_instance):
        """Installation du plugin"""
        pass
    
    def uninstall(self, erp_instance):
        """Désinstallation du plugin"""
        pass

# Exemple de plugin
class ReportingPlugin(BasePlugin):
    name = "advanced_reporting"
    version = "1.0.0"
    
    def install(self, erp_instance):
        # Ajouter des modèles de rapport
        erp_instance.register_model(ReportTemplate)
        erp_instance.register_service('reporting', ReportingService())
        
        # Ajouter des routes API
        @erp_instance.api.get("/reports/")
        def list_reports():
            return {"reports": ["sales", "inventory"]}

# Utilisation
erp = AutoERP()
plugin = ReportingPlugin()
erp.install_plugin(plugin)