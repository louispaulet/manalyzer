# Manalyzer

Manalyzer is a feminist SaaS MVP that analyzes text to determine the ratio of male to female mentions, helping to identify potential gender bias in the text.

## Features

- FastAPI based RESTful API
- Simple HTML front-end to interact with the API
- Docker and Docker Compose for easy deployment and development

## Requirements

- Docker
- Docker Compose

## Setup and Installation

1. Clone the repository:

````bash
git clone https://github.com/yourusername/manalyzer.git
````


2. Navigate to the project directory:

````bash
cd manalyzer

````

3. Build and run the Docker container using Docker Compose:

````bash
docker-compose up --build

````


This command will build the Docker image and start the container, running the FastAPI application on port 80.

4. Access the web interface by visiting `http://localhost:80` in your browser.

## API Usage

### POST /analyze

Analyze the input text and return the male-to-female mention ratio.

**Request Body**

```json
{
"text": "Sample text to analyze"
}
````

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

1. Fork the repository and create a new branch from `main` for your changes.
2. Make your changes and commit them to your branch.
3. Push your changes to your forked repository.
4. Create a pull request from your forked repository's branch to the original `main` branch.
5. Once your pull request is reviewed and approved, it will be merged into the main branch.

## License

[MIT](https://choosealicense.com/licenses/mit/)