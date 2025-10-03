# Office Apocalypse Algorithm

## Project Overview

For the complete project overview, objectives, timeline, and success criteria, see [`docs/project_overview.md`](docs/project_overview.md).

## Data Sources

The project utilizes several NYC government datasets:

- **PLUTO (Primary Land Use Tax Lot Output)**: Comprehensive property information including building classifications, areas, and valuations.
- **ACRIS (Automated City Register Information System)**: Real property legal documents and transactions.
- **DOB Permit Issuance**: Department of Buildings construction and alteration permits.
- **MTA Subway Hourly Ridership**: Public transportation usage data (2020-2024).
- **Business Registry**: Active business registrations in NYC.
- **Vacant Storefronts**: Reported vacant commercial spaces.

For detailed information on each dataset, including download links and data dictionaries, see `data/data_sources.md`.

## Dataset Integration Strategy

Each dataset captures different dimensions of office occupancy drivers:

### Dataset Roles & Relevance
- **PLUTO/MapPLUTO**: Building-level attributes (age, square footage, zoning, floors) - identifies vulnerable buildings
- **ACRIS**: Property transactions (sales, mortgages, liens) - flags distressed properties at risk
- **MTA Turnstile Data**: Subway ridership near buildings - indicates commuter demand
- **Business Registry**: Active businesses nearby - signals economic activity
- **Web-scraped Listings**: Direct vacancy evidence (days on market) - proxy for actual vacancy
- **Tax Assessment**: Property valuations and arrears - detects financial stress

### Integration Approach
All datasets center on the **commercial office building** as the unit of analysis, linked by:
- **BBL (Borough-Block-Lot)**: Primary key for property-level joins
- **Address/Geocode**: For spatial proximity analysis
- **ZIP Code**: For area-level aggregations

### Merging Process
1. **Start with PLUTO**: Universe of all NYC buildings
2. **Join ACRIS**: Add transaction history and distress signals
3. **Geospatial join MTA**: Aggregate ridership within proximity radius
4. **Join Business Data**: Count active businesses nearby
5. **Join Tax Assessment**: Add valuation and financial indicators
6. **Join Listings Data**: Label vacancy status (target variable)

This creates a comprehensive training dataset where **target = vacancy status** and **features = all other dimensions**.

## Project Structure

```
office-apocalypse-algorithm/
├── data/                    # Raw and processed datasets
│   ├── raw/                # Original downloaded files
│   ├── processed/          # Cleaned and transformed data
│   └── features/           # Engineered features
├── src/                    # Python source code
├── notebooks/              # Jupyter notebooks for analysis
├── models/                 # Saved machine learning models
├── reports/                # Generated reports and visualizations
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── docs/                   # Additional documentation
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Denis060/office-apocalypse-algorithm.git
   cd office-apocalypse-algorithm
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Exploration**: Start with notebooks in `notebooks/` to explore the datasets.

2. **Data Processing**: Run scripts in `src/` to clean and process raw data.

3. **Modeling**: Develop and train predictive models for office vacancy.

4. **Analysis**: Generate reports and visualizations in `reports/`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add license information if applicable]

## Contact

[Add contact information]