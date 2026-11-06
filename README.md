# Digital Twin for Water Distribution Networks (WDN) with Generative AI

An advanced, AI-driven digital twin framework designed to simulate, monitor, and optimize Water Distribution Networks (WDN). This project integrates traditional hydraulic modeling with Generative AI (Large Language Models and Retrieval-Augmented Generation) to provide conversational insights, anomaly detection, and predictive maintenance for critical water infrastructure.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technical Stack](#technical-stack)
- [Core Capabilities](#core-capabilities)
- [Data Pipeline & AI Integration](#data-pipeline--ai-integration)
- [Installation and Setup](#installation-and-setup)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)

## Project Overview
Managing Water Distribution Networks involves navigating complex hydraulic dynamics, detecting leaks, monitoring water quality, and predicting demand. Traditional SCADA (Supervisory Control and Data Acquisition) systems offer raw data but lack intuitive analytical interfaces. 

This project bridges that gap by creating a Digital Twin—a virtual replica of the physical WDN—augmented by Generative AI. Operators can interact with the network using natural language to query sensor data, simulate "what-if" scenarios, and receive AI-generated recommendations for pressure optimization and leak isolation.

## System Architecture
The architecture is divided into three primary layers to ensure real-time synchronization between the physical network and its digital counterpart.

1. **Hydraulic Simulation Layer:** Acts as the physical physics engine. It utilizes EPANET (or similar hydraulic solvers) to model flow, pressure, and water quality based on network topology.
2. **Data Ingestion & State Management:** Ingests simulated or real-world IoT sensor data (flow meters, pressure transducers). Time-series data is stored and processed to maintain the current state of the Digital Twin.
3. **Generative AI & Agentic Layer:** A multi-agent orchestration layer that interprets user queries, interfaces with the hydraulic model to run simulations, and analyzes the results using LLMs to formulate actionable, natural-language responses.

## Technical Stack
This framework is built using modern data engineering and AI technologies:

* **Hydraulic Modeling:** Python (via WNTR or EPANET libraries)
* **LLM Framework:** LangChain, LangGraph (for multi-agent reasoning and tool execution)
* **AI Models:** OpenAI GPT-4 / Anthropic Claude / Open-source alternatives (Llama 3)
* **Data Storage:** 
  * Time-Series Database: InfluxDB / Prometheus (for sensor data)
  * Vector Database: pgvector / Milvus (for RAG context and historical reports)
* **Backend:** FastAPI (for API endpoints and simulation triggers)
* **Infrastructure:** Docker (for containerized deployment)

## Core Capabilities
* **Conversational WDN Interface:** Ask complex questions in plain text, such as, "What happens to the pressure at Node 42 if Valve B is closed?" or "Identify the most likely source of the pressure drop in Sector 3."
* **Automated Anomaly Detection:** The system compares real-time sensor data against the baseline hydraulic model to instantly flag discrepancies that indicate leaks, pipe bursts, or faulty sensors.
* **Scenario Simulation (What-If Analysis):** The AI agent can automatically construct and execute EPANET simulation scripts based on user requests to forecast the impact of maintenance or demand spikes.
* **Knowledge Retrieval (RAG):** Integrates historical maintenance logs, asset manuals, and municipal water compliance documents, allowing the AI to ground its advice in existing protocols.

## Data Pipeline & AI Integration
To ensure the Generative AI does not hallucinate hydraulic physics, the system enforces a strict tool-calling paradigm:

1. **Query Intent:** The LLM interprets the user request.
2. **Parameter Extraction:** The LLM identifies nodes, links, timeframes, and physical metrics.
3. **Deterministic Execution:** Instead of guessing the answer, the LLM writes or triggers a Python script using the WNTR library to run a deterministic EPANET simulation.
4. **Synthesis:** The numerical output from the simulation is fed back to the LLM, which translates the raw data into a comprehensive operational report.

## Installation and Setup

### Prerequisites
* Python 3.10+
* Docker & Docker Compose (optional, for database services)
* EPANET 2.2 (included via Python dependencies)

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/adithyasr28-afk/digital-twin-wdn-genai.git](https://github.com/adithyasr28-afk/digital-twin-wdn-genai.git)
   cd digital-twin-wdn-genai
