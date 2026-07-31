"""
Kubernetes Zero-Downtime Rolling Update Automation.
Triggers seamless deployment restarts via Kubernetes API or fallback API hot-reload.
"""
import os
import time
from datetime import datetime
from typing import Optional
import requests

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


def execute_rolling_update(
    deployment_name: str = "fastapi-model-serving",
    namespace: str = "mlops",
    api_reload_url: Optional[str] = "http://localhost:8000/reload_model"
) -> bool:
    """
    Executes a Kubernetes zero-downtime rolling update on the target deployment.
    If K8s cluster is unreachable, falls back to triggering the API hot-reload endpoint.
    
    Returns:
        success: True if rollout or hot-reload succeeded.
    """
    k8s_success = False
    if KUBERNETES_AVAILABLE:
        try:
            # Try loading in-cluster config first, then kubeconfig
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()

            apps_v1 = client.AppsV1Api()
            now_iso = datetime.utcnow().isoformat()
            
            # Patch deployment annotations to trigger rolling restart
            body = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": now_iso,
                                "mlops.antigravity/model-promoted-at": now_iso
                            }
                        }
                    }
                }
            }
            
            apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            print(f"[K8s Rollout] Successfully initiated RollingUpdate for deployment '{deployment_name}' in namespace '{namespace}'.")
            k8s_success = True
        except Exception as e:
            print(f"[K8s Rollout Info] Kubernetes API unreachable or not running in cluster: {e}")

    # Fallback / Dual Hot-Reload: Notify running FastAPI service to reload promoted model
    if api_reload_url:
        try:
            print(f"[API Hot-Reload] Sending reload POST to {api_reload_url}...")
            response = requests.post(api_reload_url, timeout=5)
            if response.status_code == 200:
                print(f"[API Hot-Reload] Success: {response.json()}")
                return True
        except Exception as e:
            print(f"[API Hot-Reload] Could not reach API reload endpoint: {e}")

    return k8s_success


def simulate_k8s_rollout(deployment_name: str = "fastapi-model-serving") -> None:
    """Simulates a Kubernetes zero-downtime rolling update for CLI demos."""
    print(f"\n[K8s Simulation] Initializing RollingUpdate for '{deployment_name}'...")
    time.sleep(0.5)
    print("[K8s Simulation] Pod fastapi-model-serving-7a8f9c-new (Promoted Model) -> Status: ContainerCreating")
    time.sleep(0.5)
    print("[K8s Simulation] Pod fastapi-model-serving-7a8f9c-new (Promoted Model) -> ReadinessProbe GET /health: 200 OK")
    print("[K8s Simulation] Pod fastapi-model-serving-3b4e1f-old (Old Model)      -> Terminating gracefully (0 requests dropped)")
    print("[K8s Simulation] RollingUpdate completed successfully. Live pods switched to Promoted Model.\n")
