// code/step1_get_patients.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');

/** BEGIN MODIFICATIONS **/

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.join(outputDir, 'step1_get_patients.log');  // Changed to write log file to ../output
const logFile = fs.createWriteStream(logFilePath, { flags: 'a' });

const originalConsoleLog = console.log;
const originalConsoleError = console.error;

console.log = function (message, ...optionalParams) {
    originalConsoleLog(message, ...optionalParams);
    logFile.write('[LOG] ' + message + '\n');
    if (optionalParams.length > 0) {
        logFile.write(JSON.stringify(optionalParams) + '\n');
    }
};

console.error = function (message, ...optionalParams) {
    originalConsoleError(message, ...optionalParams);
    logFile.write('[ERROR] ' + message + '\n');
    if (optionalParams.length > 0) {
        logFile.write(JSON.stringify(optionalParams) + '\n');
    }
};

/** END MODIFICATIONS **/

/** BEGIN MODIFICATIONS **/

// Declare authToken and authTokenTimestamp at the top-level scope
let authToken;
let authTokenTimestamp;

/** END MODIFICATIONS **/

// Authenticate to get the token
async function authenticate() {
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Login Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Login Request',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: 'https://192.168.1.100:4000/login?Username=web&Password=web',
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['login'],
                            query: [
                                { key: 'Username', value: 'web' },
                                { key: 'Password', value: 'web' }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', function (err, args) {
            if (err) {
                console.error('Authentication request failed:', err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const responseJson = JSON.parse(responseBody);
                    const token = responseJson.Token;
                    resolve(token);
                } else {
                    console.error('Authentication failed with status code:', response.code);
                    reject(new Error('Authentication failed'));
                }
            }
        });
    });
}

/** BEGIN MODIFICATIONS **/

// Function to check if the token has expired
async function checkTokenExpiration() {
    const currentTime = Date.now();
    if (currentTime - authTokenTimestamp >= 60 * 60 * 1000) { // 60 minutes
        console.log('Token expired. Re-authenticating...');
        try {
            authToken = await authenticate();
            authTokenTimestamp = Date.now();
            console.log('Re-authentication successful.');
        } catch (err) {
            console.error('Failed to re-authenticate:', err);
            process.exit(1);
        }
    }
}

/** END MODIFICATIONS **/

// Function to get all patients
async function getAllPatients() {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Get Patients Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get Patients',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: 'https://192.168.1.100:4000/GET/patients',
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'patients']
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error('Request failed:', err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const filePath = path.join(outputDir, 'patients.json');
                    fs.writeFile(filePath, responseBody, function (err) {
                        if (err) {
                            console.error('Failed to save patients.json:', err);
                            reject(err);
                        } else {
                            console.log(`Patients data saved successfully to ${filePath}`);
                            resolve();
                        }
                    });
                } else if (response.code === 401) {
                    console.error('Token expired during request. Re-authenticating...');
                    try {
                        authToken = await authenticate();
                        authTokenTimestamp = Date.now();
                        console.log('Re-authentication successful. Retrying request...');
                        // Retry the request
                        await getAllPatients();
                        resolve();
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else {
                    console.error('Failed to retrieve patients. Status code:', response.code);
                    reject(new Error('Failed to retrieve patients'));
                }
            }
        });
    });
}

// Main function
(async () => {
    try {
        authToken = await authenticate();
        authTokenTimestamp = Date.now(); // Record the time the token was acquired
        console.log('Authentication successful.');
    } catch (err) {
        console.error('Failed to authenticate:', err);
        process.exit(1);
    }

    try {
        await getAllPatients();
        console.log('All patients retrieved.');
    } catch (err) {
        console.error('Failed to get patients:', err);
    }

    /** BEGIN MODIFICATIONS **/

    // Close the log file
    logFile.end();

    /** END MODIFICATIONS **/
})();