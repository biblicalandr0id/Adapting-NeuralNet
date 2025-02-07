# 1. First create an embryo generator
embryo_generator = EmbryoGenerator()

# 2. Create an agent assembler
agent_assembler = AgentAssembler(
    agent_class=AdaptiveAgent,
    birth_registry=BirthRegistry()
)

# 3. Create the development pipeline
development = EmbryoToAgentDevelopment(agent_assembler)

# 4. Create and develop an embryo into an agent
embryo = embryo_generator.create_embryo(
    parent=None,  # First generation
    position=(0, 0),
    environment={}  # Your environment settings
)

# 5. Develop embryo into agent
agent = development.develop_and_assemble(embryo)

# 6. Verify success
if agent:
    print(f"Agent created successfully!")
    print(f"Agent ID: {agent.id}")
    print(f"Genetic traits: {agent.genetic_core.get_all_traits()}")