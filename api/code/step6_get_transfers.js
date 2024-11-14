// code/step6_get_transfers.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');

/** BEGIN MODIFICATIONS **/

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/transfers');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step6_get_transfers.log');
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

// Read patients.json
const patientsFilePath = path.resolve(__dirname, '../output/patients.json');
let patientsData;
try {
    patientsData = JSON.parse(fs.readFileSync(patientsFilePath, 'utf8'));
} catch (err) {
    console.error('Failed to read patients.json:', err);
    process.exit(1);
}

const patients = patientsData.Patients;

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
                    const authToken = responseJson.Token;
                    resolve(authToken);
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

// Function to get transfers for a patient
async function getTransfers(patientIDx) {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Get Transfers Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get Transfers',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/transfers?PatientIDx=${encodeURIComponent(patientIDx)}`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'transfers'],
                            query: [
                                { key: 'PatientIDx', value: patientIDx }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error(`Request failed for PatientIDx ${patientIDx}:`, err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const transfersData = JSON.parse(responseBody);
                    const filePath = path.join(outputDir, `transfers_${sanitizeFilename(patientIDx)}.json`);
                    fs.writeFile(filePath, responseBody, function (err) {
                        if (err) {
                            console.error(`Failed to save transfers for PatientIDx ${patientIDx}:`, err);
                            reject(err);
                        } else {
                            console.log(`Transfers for PatientIDx ${patientIDx} saved successfully.`);
                            resolve(transfersData);
                        }
                    });
                } else if (response.code === 401) {
                    console.error('Token expired during request. Re-authenticating...');
                    try {
                        authToken = await authenticate();
                        authTokenTimestamp = Date.now();
                        console.log('Re-authentication successful. Retrying request...');
                        // Retry the request
                        const data = await getTransfers(patientIDx);
                        resolve(data);
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    console.log(`No transfers found for PatientIDx ${patientIDx}.`);
                    resolve(null);
                } else {
                    console.error(`Error retrieving transfers for PatientIDx ${patientIDx}: Status code ${response.code}`);
                    reject(new Error(`Status code ${response.code}`));
                }
            }
        });
    });
}

// Function to get transfer info
async function getTransferInfo(transferID) {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Get TransferInfo Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get TransferInfo',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/transferinfo?TransferID=${encodeURIComponent(transferID)}`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'transferinfo'],
                            query: [
                                { key: 'TransferID', value: transferID }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error(`Request failed for TransferID ${transferID}:`, err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const filePath = path.join(outputDir, `transferinfo_${sanitizeFilename(transferID)}.json`);
                    fs.writeFile(filePath, responseBody, function (err) {
                        if (err) {
                            console.error(`Failed to save transfer info for TransferID ${transferID}:`, err);
                            reject(err);
                        } else {
                            console.log(`Transfer info for TransferID ${transferID} saved successfully.`);
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
                        await getTransferInfo(transferID);
                        resolve();
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    console.log(`No transfer info found for TransferID ${transferID}.`);
                    resolve();
                } else {
                    console.error(`Error retrieving transfer info for TransferID ${transferID}: Status code ${response.code}`);
                    reject(new Error(`Status code ${response.code}`));
                }
            }
        });
    });
}

/** BEGIN MODIFICATIONS **/

// Function to sanitize filenames
function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9]/gi, '_');
}

// Delay function to enforce API rate limit
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/** END MODIFICATIONS **/

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

    /** BEGIN MODIFICATIONS **/

    // Set initial timestamp for rate limiting
    let lastRequestTime = Date.now();

    /** END MODIFICATIONS **/

    // Iterate over patients and get transfers
    for (const patient of patients) {
        const patientIDx = patient.PatientIDx;
        console.log(`Processing PatientIDx: ${patientIDx}`);

        /** BEGIN MODIFICATIONS **/

        // Enforce API rate limit of 10 requests per second
        const currentTime = Date.now();
        const timeSinceLastRequest = currentTime - lastRequestTime;
        if (timeSinceLastRequest < 100) {
            await delay(100 - timeSinceLastRequest);
        }
        lastRequestTime = Date.now();

        /** END MODIFICATIONS **/

        try {
            const transfersData = await getTransfers(patientIDx);

            if (transfersData && transfersData.TransferIDList && transfersData.TransferIDList.length > 0) {
                // Iterate over Transfer IDs and get transfer info
                for (const transferID of transfersData.TransferIDList) {
                    console.log(`  Processing TransferID: ${transferID}`);

                    /** BEGIN MODIFICATIONS **/

                    // Enforce API rate limit of 10 requests per second
                    const currentTime = Date.now();
                    const timeSinceLastRequest = currentTime - lastRequestTime;
                    if (timeSinceLastRequest < 100) {
                        await delay(100 - timeSinceLastRequest);
                    }
                    lastRequestTime = Date.now();

                    /** END MODIFICATIONS **/

                    try {
                        await getTransferInfo(transferID);
                    } catch (err) {
                        console.error(`Error processing TransferID ${transferID}:`, err);
                    }
                }
            }
        } catch (err) {
            console.error(`Error processing PatientIDx ${patientIDx}:`, err);
        }
    }

    console.log('All transfers and transfer info retrieved.');

    /** BEGIN MODIFICATIONS **/

    // Close the log file
    logFile.end();

    /** END MODIFICATIONS **/
})();

/** BEGIN MODIFICATIONS **/

// Declare authToken and authTokenTimestamp at the top-level scope
let authToken;
let authTokenTimestamp;

/** END MODIFICATIONS **/