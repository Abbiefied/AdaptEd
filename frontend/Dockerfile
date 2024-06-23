# Dockerfile

# Use an official Node.js runtime as a parent image
FROM node:18

# Set the working directory
WORKDIR /app

# Clear npm cache
RUN npm cache clean --force

# Copy the package.json and package-lock.json
COPY package*.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the React app
RUN npm run build

# Install serve to serve the build folder
RUN npm install -g serve

# Command to run the app
CMD ["serve", "-s", "build", "-l", "3000"]
