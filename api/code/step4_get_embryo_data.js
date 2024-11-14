// code/step4_get_embryo_data.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');

/** BEGIN MODIFICATIONS **/

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/embryo_data');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step4_get_embryo_data.log');
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

// Read embryo IDs from the embryo_ids directory
const embryoIDsDir = path.resolve(__dirname, '../output/embryo_ids');
let embryoIDFiles = fs.readdirSync(embryoIDsDir).filter(file => file.endsWith('.json'));

let embryoIDs = [];

for (const file of embryoIDFiles) {
    const filePath = path.join(embryoIDsDir, file);
    let data;
    try {
        data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (err) {
        console.error(`Failed to read ${file}:`, err);
        continue;
    }

    if (data.EmbryoIDList && data.EmbryoIDList.length > 0) {
        embryoIDs.push(...data.EmbryoIDList);
    }
}

if (embryoIDs.length === 0) {
    console.error('No embryo IDs found.');
    process.exit(1);
}

// Remove duplicate embryo IDs
embryoIDs = [...new Set(embryoIDs)];

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

// Function to get embryo data for an embryo ID
async function getEmbryoData(embryoID) {
    await checkTokenExpiration();

    const embryoData = { EmbryoID: embryoID };

    // Define API requests
    const requests = [
        {
            name: 'Get Embryo Details',
            endpoint: 'embryodetails',
            dataKey: 'EmbryoDetails',
            params: { EmbryoID: embryoID }
        },
        {
            name: 'Get Annotations',
            endpoint: 'annotation',
            dataKey: 'AnnotationList',
            params: { EmbryoID: embryoID }
        },
        {
            name: 'Get Evaluation',
            endpoint: 'evaluation',
            dataKey: 'Evaluation',
            params: { EmbryoID: embryoID }
        },
        {
            name: 'Get Embryo Fate',
            endpoint: 'embryofate',
            dataKey: 'EmbryoFate',
            params: { EmbryoID: embryoID }
        }
        // Add more requests here if needed
    ];

    for (const request of requests) {
        try {
            /** BEGIN MODIFICATIONS **/

            // Enforce API rate limit of 10 requests per second
            const currentTime = Date.now();
            const timeSinceLastRequest = currentTime - getEmbryoData.lastRequestTime;
            if (timeSinceLastRequest < 100) { // 100 ms between requests
                await delay(100 - timeSinceLastRequest);
            }
            getEmbryoData.lastRequestTime = Date.now();

            /** END MODIFICATIONS **/

            const data = await makeApiRequest(request.endpoint, request.params);
            embryoData[request.dataKey] = data;
        } catch (err) {
            console.error(`Error retrieving ${request.name} for EmbryoID ${embryoID}:`, err);
        }
    }

    // Save embryo data to file
    const sanitizedEmbryoID = sanitizeFilename(embryoID);
    const filePath = path.join(outputDir, `embryo_data_${sanitizedEmbryoID}.json`);
    fs.writeFileSync(filePath, JSON.stringify(embryoData, null, 2));
    console.log(`Embryo data saved for EmbryoID ${embryoID}.`);
}

// Initialize lastRequestTime for rate limiting
getEmbryoData.lastRequestTime = Date.now();

// Function to make API requests
async function makeApiRequest(endpoint, params) {
    await checkTokenExpiration();

    return new Promise((resolve, reject) => {
        const queryParameters = Object.entries(params).map(([key, value]) => ({
            key: key,
            value: value
        }));

        newman.run({
            collection: {
                info: {
                    name: `Get ${endpoint} Collection`,
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: `Get ${endpoint}`,
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/${endpoint}?${new URLSearchParams(params).toString()}`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', endpoint],
                            query: queryParameters
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const responseJson = JSON.parse(responseBody);
                    resolve(responseJson);
                } else if (response.code === 401) {
                    console.error('Token expired during request. Re-authenticating...');
                    try {
                        authToken = await authenticate();
                        authTokenTimestamp = Date.now();
                        console.log('Re-authentication successful. Retrying request...');
                        // Retry the request
                        const data = await makeApiRequest(endpoint, params);
                        resolve(data);
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    resolve(null); // No data found
                } else {
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

    // Set initial timestamp
    let lastEmbryoProcessingTime = Date.now();

    /** END MODIFICATIONS **/

    // Iterate over embryo IDs
    for (const embryoID of embryoIDs) {
        console.log(`Processing EmbryoID: ${embryoID}`);

        /** BEGIN MODIFICATIONS **/

        // Enforce API rate limit of 10 requests per second between embryos
        const currentTime = Date.now();
        const timeSinceLastEmbryo = currentTime - lastEmbryoProcessingTime;
        if (timeSinceLastEmbryo < 100) { // 100 ms between embryo processing
            await delay(100 - timeSinceLastEmbryo);
        }
        lastEmbryoProcessingTime = Date.now();

        /** END MODIFICATIONS **/

        try {
            await getEmbryoData(embryoID);
        } catch (err) {
            console.error(`Error processing EmbryoID ${embryoID}:`, err);
        }
    }

    console.log('All embryo data retrieved.');

    /** BEGIN MODIFICATIONS **/

    // Close the log file
    logFile.end();

    /** END MODIFICATIONS **/
})();