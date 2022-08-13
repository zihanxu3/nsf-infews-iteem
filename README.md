# nsf-infews-iteem

Project Description: This project is mainly using React for frontend and Flask for backend. Both the two ends' servers are now deployed on the University's cPanel service.

# File Structure:
## Frontend
```
.
├── public        
├── node_modules
└── src               [Source files for frontend]
    ├── assets        [Assets (images etc)]
    └── components    [Pages, components for pages]
```
## Backend
```
.
├── ITEEM.py          [ITEEM model]
├── LICENSE
├── README.md
├── api.py            [Main file for backend logic]
└── requirements.txt  [For deployemnt]
```

# For frontend (Development Environment Setup)

## Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

### Available Scripts

In the project directory, you can run:

#### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

#### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

#### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

#### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

### Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

#### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

#### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

#### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

#### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

#### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

#### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)


# For frontend (Deployment Environment Setup)

Refer to this article: https://dev.to/crishanks/deploy-host-your-react-app-with-cpanel-in-under-5-minutes-4mf6


# For backend (Development Environment Setup) 

According to your own requirements, you might need to install the necessary python packages beforehand. I used pip to install all the packages that I need and did not use a virtual environment (even though that’s highly recommended). To create a new python virtual environment called venv (you can call this something else — just replace the last venv in the command below with your own venv name), 
run: `python -m venv venv`

To activate venv:

`source venv/bin/activate`

Refer: https://towardsdatascience.com/build-deploy-a-react-flask-app-47a89a5d17d9
Flask Docs: https://flask.palletsprojects.com/en/1.1.x/quickstart/

# For backend (Deployment Environment Setup)

Refer to this article: https://skilllx.com/how-to-host-flask-application-on-cpanel/

# For plotting

The plots of this website consist mainly of two tools, leaflet (react-leaflet: https://react-leaflet.js.org) and highcharts (https://github.com/highcharts/highcharts-react). Please refer to these documentations for plotting-related problems.

# For UI Framework

The UI framwork for this project is material UI, please refer to https://mui.com for documentation.

Email zihanxu3(at)illinois(dot)edu for any questions not answered above. 
