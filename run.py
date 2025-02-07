import os
from embryo_generator import EmbryoGenerator, EmbryoToAgentDevelopment
from agent_assembly import AgentAssembler
from birth_records import BirthRegistry
from agent import AdaptiveAgent
from logging import getLogger, basicConfig, INFO
from datetime import datetime

# Setup logging
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

def main():
    # Create output directories if they don't exist
    for dir_path in ['agents', 'logs', 'genetic_records', 'development_logs']:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        # Initialize core components
        birth_registry = BirthRegistry()
        agent_assembler = AgentAssembler(
            agent_class=AdaptiveAgent,
            birth_registry=birth_registry
        )
        embryo_generator = EmbryoGenerator()
        development_pipeline = EmbryoToAgentDevelopment(agent_assembler)

        # Create initial embryo
        logger.info("Creating embryo...")
        embryo = embryo_generator.create_embryo(
            parent=None,
            position=(0, 0),
            environment={"time": datetime.now().isoformat()}
        )
        
        # Develop and assemble agent
        logger.info("Developing embryo into agent...")
        agent = development_pipeline.develop_and_assemble(embryo)

        if agent:
            logger.info(f"Successfully created agent with ID: {agent.id}")
            logger.info("Genetic Traits:")
            for category, traits in agent.genetic_core.get_all_traits().items():
                logger.info(f"{category}: {traits}")
        else:
            logger.error("Failed to create agent")

    except Exception as e:
        logger.error(f"Error during agent creation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()