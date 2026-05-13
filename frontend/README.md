# Frontend Client - Insurance Fraud Detection

This directory contains the user interface for the Insurance Fraud Detection system. Built as a Single Page Application (SPA), it provides intuitive dashboards for risk analysis, claim management, and user portals.

## Tech Stack

- **Framework**: [React 19](https://react.dev/)
- **Language**: TypeScript
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Routing**: React Router DOM
- **HTTP Client**: Axios
- **Form Validation**: Zod
- **Icons**: Lucide React
- **Web Server (Prod)**: Nginx

## Prerequisites

To run this application locally, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- `npm` (Node Package Manager)

## Local Development Setup

1. **Install Dependencies**
   Navigate to the frontend directory and install the required npm packages.

   ```bash
   cd frontend
   npm install
   ```

2. **Environment Configuration**
   Ensure your application knows how to communicate with the backend API. Create a `.env` file based on any provided examples, or ensure the following variable is set:

   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

3. **Start the Development Server**
   Spin up the Vite development server with Hot Module Replacement (HMR).
   ```bash
   npm run dev
   ```
   The application will be accessible at `http://localhost:5173`.

## Available Scripts

In the project directory, you can run:

- `npm run dev`: Starts the local development server.
- `npm run build`: Compiles TypeScript and builds the app for production into the `dist/` folder.
- `npm run lint`: Runs ESLint to catch syntax and style issues.
- `npm run preview`: Bootstraps a local web server to preview the production build generated in the `dist/` folder.

## Docker Deployment

The frontend includes a production-ready `Dockerfile` that executes a multi-stage build:

1. **Build Stage**: Compiles the React/Vite application into static assets.
2. **Runtime Stage**: Uses an Nginx alpine image to serve the compiled assets, utilizing the custom `nginx.conf` provided in the repository for optimal routing (handling SPA client-side routing).

To build and run the frontend independently:

```bash
docker build -t insurance-fraud-frontend .
docker run -p 5173:80 insurance-fraud-frontend
```
