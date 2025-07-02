[Skip to main content](https://docs.openwebui.com/getting-started/advanced-topics/development/#__docusaurus_skipToContent_fallback)

On this page

Welcome to the **Open WebUI Development Setup Guide!** Whether you're a novice or an experienced developer, this guide will help you set up a **local development environment** for both the frontend and backend components. Let’s dive in! 🚀

## System Requirements [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#system-requirements "Direct link to System Requirements")

- **Operating System**: Linux (or WSL on Windows) or macOS
- **Python Version**: Python 3.11+
- **Node.js Version**: 22.10+

## Development Methods [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#development-methods "Direct link to Development Methods")

### 🐧 Local Development Setup [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#-local-development-setup "Direct link to 🐧 Local Development Setup")

1. **Clone the Repository**:





```codeBlockLines_e6Vv
git clone https://github.com/open-webui/open-webui.git
cd open-webui

```

2. **Frontend Setup**:
   - Create a `.env` file:





     ```codeBlockLines_e6Vv
     cp -RPp .env.example .env

     ```

   - Install dependencies:





     ```codeBlockLines_e6Vv
     npm install

     ```

   - Start the frontend server:





     ```codeBlockLines_e6Vv
     npm run dev

     ```









     🌐 Available at: [http://localhost:5173](http://localhost:5173/).
3. **Backend Setup**:
   - Navigate to the backend:





     ```codeBlockLines_e6Vv
     cd backend

     ```

   - Use **Conda** for environment setup:





     ```codeBlockLines_e6Vv
     conda create --name open-webui python=3.11
     conda activate open-webui

     ```

   - Install dependencies:





     ```codeBlockLines_e6Vv
     pip install -r requirements.txt -U

     ```

   - Start the backend:





     ```codeBlockLines_e6Vv
     sh dev.sh

     ```









     📄 API docs available at: [http://localhost:8080/docs](http://localhost:8080/docs).

## 🐛 Troubleshooting [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#-troubleshooting "Direct link to 🐛 Troubleshooting")

### **FATAL ERROR: Reached Heap Limit** [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#fatal-error-reached-heap-limit "Direct link to fatal-error-reached-heap-limit")

If you encounter memory-related errors during the build, increase the **Node.js heap size**:

1. **Modify Dockerfile**:





```codeBlockLines_e6Vv
ENV NODE_OPTIONS=--max-old-space-size=4096

```

2. **Allocate at least 4 GB of RAM** to Node.js.


* * *

### **Other Issues** [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#other-issues "Direct link to other-issues")

- **Port Conflicts**:

Ensure that no other processes are using **ports 8080 or 5173**.

- **Hot Reload Not Working**:

Verify that **watch mode** is enabled for both frontend and backend.


## Contributing to Open WebUI [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#contributing-to-open-webui "Direct link to Contributing to Open WebUI")

### Local Workflow [​](https://docs.openwebui.com/getting-started/advanced-topics/development/\#local-workflow "Direct link to Local Workflow")

1. **Commit Changes Regularly** to track progress.

2. **Sync with the Main Branch** to avoid conflicts:





```codeBlockLines_e6Vv
git pull origin main

```

3. **Run Tests Before Pushing**:





```codeBlockLines_e6Vv
npm run test

```


Happy coding! 🎉

- [System Requirements](https://docs.openwebui.com/getting-started/advanced-topics/development/#system-requirements)
- [Development Methods](https://docs.openwebui.com/getting-started/advanced-topics/development/#development-methods)
  - [🐧 Local Development Setup](https://docs.openwebui.com/getting-started/advanced-topics/development/#-local-development-setup)
- [🐛 Troubleshooting](https://docs.openwebui.com/getting-started/advanced-topics/development/#-troubleshooting)
  - [**FATAL ERROR: Reached Heap Limit**](https://docs.openwebui.com/getting-started/advanced-topics/development/#fatal-error-reached-heap-limit)
  - [**Other Issues**](https://docs.openwebui.com/getting-started/advanced-topics/development/#other-issues)
- [Contributing to Open WebUI](https://docs.openwebui.com/getting-started/advanced-topics/development/#contributing-to-open-webui)
  - [Local Workflow](https://docs.openwebui.com/getting-started/advanced-topics/development/#local-workflow)