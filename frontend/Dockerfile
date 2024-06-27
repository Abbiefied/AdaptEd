FROM node:18

WORKDIR /app

RUN npm cache clean --force

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

RUN npm install -g serve

CMD ["serve", "-s", "build", "-l", "3000"]
