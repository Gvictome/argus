"""
Industry Concentrations - Domain-specific configurations for agents

Each industry has:
- Recommended agents
- Agent buffs (enhancements)
- Default plugins
- Specialized prompts/behaviors
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Industry(Enum):
    """Supported industry concentrations"""
    # Tech
    SECURITY = "security"           # Home security, surveillance
    IOT = "iot"                     # Internet of Things, smart home
    WEB = "web"                     # Web development
    MOBILE = "mobile"               # Mobile app development
    AI_ML = "ai_ml"                 # AI/Machine Learning
    DEVOPS = "devops"               # DevOps, infrastructure
    BLOCKCHAIN = "blockchain"       # Blockchain, crypto
    GAMING = "gaming"               # Game development

    # Business
    ECOMMERCE = "ecommerce"         # E-commerce, retail
    FINTECH = "fintech"             # Financial technology
    HEALTHCARE = "healthcare"       # Healthcare, medical
    EDUCATION = "education"         # EdTech, learning
    MARKETING = "marketing"         # Marketing, advertising
    SAAS = "saas"                   # SaaS products

    # Hardware
    ROBOTICS = "robotics"           # Robotics, automation
    EMBEDDED = "embedded"           # Embedded systems
    RASPBERRY_PI = "raspberry_pi"   # Raspberry Pi projects

    # General
    GENERAL = "general"             # General purpose


@dataclass
class AgentBuff:
    """
    Enhancement applied to an agent based on context

    Buffs modify agent behavior, add capabilities, or
    adjust parameters for specific industries/tasks.
    """
    name: str
    description: str
    modifiers: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    priority_boost: int = 0  # -10 to +10
    enabled: bool = True


@dataclass
class IndustryConfig:
    """Configuration for an industry concentration"""
    name: str
    description: str
    recommended_agents: List[str]
    recommended_plugins: List[str]
    agent_buffs: Dict[str, List[AgentBuff]]  # agent_type -> buffs
    default_settings: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)  # For auto-detection


# ============================================================================
# Industry Configurations
# ============================================================================

INDUSTRY_CONFIGS: Dict[Industry, IndustryConfig] = {

    Industry.SECURITY: IndustryConfig(
        name="Security & Surveillance",
        description="Home security, surveillance systems, access control",
        recommended_agents=["camera", "detection", "hardware", "automation", "error"],
        recommended_plugins=["camera", "detection", "automation", "telegram", "scheduler"],
        agent_buffs={
            "camera": [
                AgentBuff(
                    name="surveillance_mode",
                    description="Optimized for 24/7 monitoring",
                    modifiers={"fps": 15, "motion_sensitivity": "high", "night_mode": True},
                    capabilities=["motion_detection", "night_vision", "continuous_recording"]
                ),
                AgentBuff(
                    name="alert_priority",
                    description="Prioritize security alerts",
                    priority_boost=5
                )
            ],
            "detection": [
                AgentBuff(
                    name="security_detection",
                    description="Focus on humans, vehicles, weapons",
                    modifiers={"priority_classes": ["person", "car", "truck", "knife"]},
                    capabilities=["face_recognition", "intrusion_detection", "loitering_detection"]
                )
            ],
            "automation": [
                AgentBuff(
                    name="security_automation",
                    description="Security-focused automations",
                    capabilities=["lock_control", "alarm_trigger", "light_deterrent"]
                )
            ]
        },
        default_settings={
            "alert_threshold": "low",
            "recording_mode": "motion",
            "retention_days": 30
        },
        keywords=["security", "surveillance", "camera", "alarm", "intrusion", "monitoring"]
    ),

    Industry.IOT: IndustryConfig(
        name="IoT & Smart Home",
        description="Smart home devices, IoT sensors, home automation",
        recommended_agents=["automation", "hardware", "network", "error"],
        recommended_plugins=["automation", "mqtt", "homeassistant", "scheduler"],
        agent_buffs={
            "automation": [
                AgentBuff(
                    name="smart_home",
                    description="Smart home optimization",
                    modifiers={"protocol": "mqtt", "discovery": True},
                    capabilities=["device_discovery", "scene_control", "energy_monitoring"]
                )
            ],
            "hardware": [
                AgentBuff(
                    name="sensor_focus",
                    description="Optimized for sensor management",
                    capabilities=["i2c_scanning", "gpio_management", "sensor_polling"]
                )
            ],
            "network": [
                AgentBuff(
                    name="iot_network",
                    description="IoT network management",
                    capabilities=["mqtt_broker", "device_discovery", "network_mapping"]
                )
            ]
        },
        default_settings={
            "mqtt_enabled": True,
            "auto_discovery": True
        },
        keywords=["iot", "smart home", "automation", "sensor", "mqtt", "zigbee"]
    ),

    Industry.AI_ML: IndustryConfig(
        name="AI & Machine Learning",
        description="AI models, machine learning, computer vision",
        recommended_agents=["detection", "error", "dependency"],
        recommended_plugins=["detection", "database", "scheduler"],
        agent_buffs={
            "detection": [
                AgentBuff(
                    name="ml_pipeline",
                    description="Full ML pipeline support",
                    modifiers={"model_format": "tflite", "quantization": True},
                    capabilities=["model_training", "inference_optimization", "batch_processing"]
                )
            ],
            "error": [
                AgentBuff(
                    name="ml_errors",
                    description="ML-specific error handling",
                    capabilities=["cuda_errors", "memory_management", "model_debugging"]
                )
            ],
            "dependency": [
                AgentBuff(
                    name="ml_deps",
                    description="ML dependency expertise",
                    modifiers={"priority_packages": ["tensorflow", "pytorch", "numpy", "opencv"]},
                    capabilities=["gpu_detection", "version_compatibility"]
                )
            ]
        },
        default_settings={
            "gpu_enabled": True,
            "model_cache": True
        },
        keywords=["ai", "ml", "machine learning", "tensorflow", "pytorch", "model"]
    ),

    Industry.WEB: IndustryConfig(
        name="Web Development",
        description="Web applications, APIs, frontend/backend",
        recommended_agents=["error", "dependency", "network", "syntax"],
        recommended_plugins=["database", "docker", "git"],
        agent_buffs={
            "error": [
                AgentBuff(
                    name="web_errors",
                    description="Web-specific error handling",
                    capabilities=["cors_fixing", "api_debugging", "frontend_errors"]
                )
            ],
            "network": [
                AgentBuff(
                    name="web_network",
                    description="Web server management",
                    capabilities=["port_management", "ssl_setup", "proxy_config"]
                )
            ],
            "syntax": [
                AgentBuff(
                    name="web_syntax",
                    description="Web language support",
                    modifiers={"languages": ["javascript", "typescript", "html", "css", "python"]},
                    capabilities=["jsx_support", "async_patterns"]
                )
            ]
        },
        default_settings={
            "framework": "fastapi",
            "hot_reload": True
        },
        keywords=["web", "api", "frontend", "backend", "react", "fastapi", "django"]
    ),

    Industry.RASPBERRY_PI: IndustryConfig(
        name="Raspberry Pi",
        description="Raspberry Pi projects, GPIO, embedded Linux",
        recommended_agents=["hardware", "camera", "error", "network"],
        recommended_plugins=["camera", "automation", "scheduler"],
        agent_buffs={
            "hardware": [
                AgentBuff(
                    name="pi_hardware",
                    description="Raspberry Pi hardware expertise",
                    modifiers={"platform": "raspberry_pi", "gpio_library": "gpiozero"},
                    capabilities=["gpio_control", "i2c_management", "spi_control", "pwm"]
                )
            ],
            "camera": [
                AgentBuff(
                    name="picamera",
                    description="Pi Camera optimization",
                    modifiers={"library": "picamera2", "format": "h264"},
                    capabilities=["csi_camera", "libcamera", "hardware_encoding"]
                )
            ],
            "error": [
                AgentBuff(
                    name="pi_errors",
                    description="Pi-specific troubleshooting",
                    capabilities=["kernel_errors", "driver_issues", "power_problems"]
                )
            ]
        },
        default_settings={
            "platform": "raspberry_pi",
            "headless": True
        },
        keywords=["raspberry pi", "rpi", "gpio", "picamera", "embedded", "linux"]
    ),

    Industry.DEVOPS: IndustryConfig(
        name="DevOps & Infrastructure",
        description="CI/CD, containers, deployment, infrastructure",
        recommended_agents=["error", "network", "dependency"],
        recommended_plugins=["docker", "git", "scheduler"],
        agent_buffs={
            "error": [
                AgentBuff(
                    name="devops_errors",
                    description="Infrastructure error handling",
                    capabilities=["container_debugging", "ci_failures", "deployment_issues"]
                )
            ],
            "network": [
                AgentBuff(
                    name="infra_network",
                    description="Infrastructure networking",
                    capabilities=["load_balancing", "dns_management", "firewall_config"]
                )
            ]
        },
        default_settings={
            "containerized": True,
            "ci_cd": True
        },
        keywords=["devops", "docker", "kubernetes", "ci/cd", "deployment", "infrastructure"]
    ),

    Industry.GENERAL: IndustryConfig(
        name="General Purpose",
        description="General software development",
        recommended_agents=["error", "syntax", "dependency"],
        recommended_plugins=["git", "database"],
        agent_buffs={
            "error": [
                AgentBuff(
                    name="general_errors",
                    description="Broad error handling",
                    capabilities=["multi_language", "stack_trace_analysis"]
                )
            ]
        },
        default_settings={},
        keywords=[]
    ),
}


class IndustryDetector:
    """Auto-detect industry from project content"""

    @staticmethod
    def detect(project_path: str, project_name: str = "", description: str = "") -> Industry:
        """
        Detect industry from project context

        Args:
            project_path: Path to project
            project_name: Project name
            description: Project description

        Returns:
            Best matching Industry
        """
        text = f"{project_name} {description}".lower()

        # Check each industry's keywords
        scores: Dict[Industry, int] = {}

        for industry, config in INDUSTRY_CONFIGS.items():
            score = 0
            for keyword in config.keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                scores[industry] = score

        if scores:
            return max(scores, key=scores.get)

        return Industry.GENERAL

    @staticmethod
    def get_config(industry: Industry) -> IndustryConfig:
        """Get configuration for an industry"""
        return INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS[Industry.GENERAL])


class AgentEnhancer:
    """
    Applies buffs to agents based on context

    The enhancer modifies agent capabilities and behavior
    based on project industry, current task, and conditions.
    """

    def __init__(self):
        self.active_buffs: Dict[str, List[AgentBuff]] = {}

    def get_buffs_for_agent(
        self,
        agent_type: str,
        industry: Industry,
        task: Optional[str] = None
    ) -> List[AgentBuff]:
        """
        Get all applicable buffs for an agent

        Args:
            agent_type: Type of agent
            industry: Project industry
            task: Current task (optional)

        Returns:
            List of applicable buffs
        """
        config = IndustryDetector.get_config(industry)
        buffs = config.agent_buffs.get(agent_type, [])

        # Filter enabled buffs
        active = [b for b in buffs if b.enabled]

        # Task-specific buffs could be added here
        if task:
            # Could add task-specific buff logic
            pass

        return active

    def apply_buffs(
        self,
        agent_type: str,
        agent_config: Dict[str, Any],
        industry: Industry,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply buffs to agent configuration

        Args:
            agent_type: Type of agent
            agent_config: Base agent configuration
            industry: Project industry
            task: Current task

        Returns:
            Enhanced configuration
        """
        buffs = self.get_buffs_for_agent(agent_type, industry, task)
        enhanced = agent_config.copy()

        for buff in buffs:
            # Apply modifiers
            enhanced.update(buff.modifiers)

            # Add capabilities
            if "capabilities" not in enhanced:
                enhanced["capabilities"] = []
            enhanced["capabilities"].extend(buff.capabilities)

            # Apply priority boost
            current_priority = enhanced.get("priority", 5)
            enhanced["priority"] = max(1, min(10, current_priority + buff.priority_boost))

        # Track active buffs
        self.active_buffs[agent_type] = buffs

        return enhanced

    def get_enhanced_spawn_config(
        self,
        agent_type: str,
        industry: Industry,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get full spawn configuration for an agent with buffs applied

        Args:
            agent_type: Type of agent to spawn
            industry: Project industry
            task: Current task

        Returns:
            Complete spawn configuration
        """
        # Base config
        config = {
            "type": agent_type,
            "industry": industry.value,
            "priority": 5,
            "capabilities": [],
            "modifiers": {}
        }

        # Apply buffs
        enhanced = self.apply_buffs(agent_type, config, industry, task)

        # Add industry defaults
        industry_config = IndustryDetector.get_config(industry)
        enhanced["industry_settings"] = industry_config.default_settings

        return enhanced

    def list_active_buffs(self) -> Dict[str, List[str]]:
        """List all currently active buffs by agent"""
        return {
            agent: [b.name for b in buffs]
            for agent, buffs in self.active_buffs.items()
        }


# Singleton instances
industry_detector = IndustryDetector()
agent_enhancer = AgentEnhancer()
