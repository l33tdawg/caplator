"""Runtime hook to configure Qt plugin paths."""
import os
import sys

# Set Qt plugin path before PyQt6 is imported
if getattr(sys, 'frozen', False):
    # Running as bundled app
    bundle_dir = os.path.dirname(sys.executable)
    
    # For .app bundle, we need to go up to Resources
    if bundle_dir.endswith('MacOS'):
        resources_dir = os.path.join(os.path.dirname(bundle_dir), 'Resources')
        if os.path.exists(resources_dir):
            bundle_dir = resources_dir
    
    qt_plugins = os.path.join(bundle_dir, 'PyQt6', 'Qt6', 'plugins')
    
    if os.path.exists(qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins
    
    # Also set QT_QPA_PLATFORM_PLUGIN_PATH for platforms specifically
    platforms_dir = os.path.join(qt_plugins, 'platforms')
    if os.path.exists(platforms_dir):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platforms_dir
