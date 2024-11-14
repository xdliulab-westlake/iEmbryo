// code/step5_get_all_images.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');
const extract = require('extract-zip');

/** BEGIN MODIFICATIONS **/

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/images');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step5_get_all_images.log');
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

// Function to authenticate and handle token expiration
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

// Function to get all images for an embryo ID
async function getAllImages(embryoID) {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        const embryoDir = path.join(outputDir, sanitizeFilename(embryoID));
        if (!fs.existsSync(embryoDir)) {
            fs.mkdirSync(embryoDir, { recursive: true });
        }

        newman.run({
            collection: {
                info: {
                    name: 'Get All Images Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get All Images',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/allimages?EmbryoID=${encodeURIComponent(embryoID)}&ImageOverlay=true`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'allimages'],
                            query: [
                                { key: 'EmbryoID', value: embryoID },
                                { key: 'ImageOverlay', value: 'true' }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error(`Request failed for EmbryoID ${embryoID}:`, err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    // Save the ZIP file
                    const contentTypeHeader = response.headers.find(header => header.key.toLowerCase() === 'content-type');
                    const contentType = contentTypeHeader ? contentTypeHeader.value.split(';')[0].trim() : null;
                    if (contentType && (contentType === 'application/zip' || contentType === 'images/zip')) {
                        const zipBuffer = response.stream;
                        const zipFilePath = path.join(embryoDir, 'images.zip');
                        fs.writeFile(zipFilePath, zipBuffer, function (err) {
                            if (err) {
                                console.error(`Failed to save images for EmbryoID ${embryoID}:`, err);
                                reject(err);
                            } else {
                                console.log(`Images for EmbryoID ${embryoID} saved successfully.`);
                                // Extract the ZIP file
                                extractZip(zipFilePath, embryoDir).then(() => {
                                    // Delete the ZIP file after extraction
                                    fs.unlinkSync(zipFilePath);
                                    resolve();
                                }).catch(err => {
                                    console.error(`Failed to extract images for EmbryoID ${embryoID}:`, err);
                                    reject(err);
                                });
                            }
                        });
                    } else {
                        console.error(`Unexpected Content-Type for EmbryoID ${embryoID}:`, contentType);
                        reject(new Error('Unexpected Content-Type'));
                    }
                } else if (response.code === 401) {
                    console.error('Token expired during request. Re-authenticating...');
                    try {
                        authToken = await authenticate();
                        authTokenTimestamp = Date.now();
                        console.log('Re-authentication successful. Retrying request...');
                        // Retry the request
                        await getAllImages(embryoID);
                        resolve();
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    console.log(`No images found for EmbryoID ${embryoID}.`);
                    resolve();
                } else {
                    console.error(`Error retrieving images for EmbryoID ${embryoID}: Status code ${response.code}`);
                    reject(new Error(`Status code ${response.code}`));
                }
            }
        });
    });
}

// Function to extract ZIP file
function extractZip(zipFilePath, outputDir) {
    return extract(zipFilePath, { dir: outputDir });
}

// Function to sanitize filenames
function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9]/gi, '_');
}

/** BEGIN MODIFICATIONS **/

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
    let lastRequestTime = Date.now();

    /** END MODIFICATIONS **/

    // Iterate over embryo IDs
    for (const embryoID of embryoIDs) {
        console.log(`Processing EmbryoID: ${embryoID}`);

        /** BEGIN MODIFICATIONS **/

        // Enforce API rate limit of 10 requests per second
        const currentTime = Date.now();
        const timeSinceLastRequest = currentTime - lastRequestTime;
        if (timeSinceLastRequest < 100) { // 100 ms between requests
            await delay(100 - timeSinceLastRequest);
        }
        lastRequestTime = Date.now();

        /** END MODIFICATIONS **/

        try {
            await getAllImages(embryoID);
        } catch (err) {
            console.error(`Error processing EmbryoID ${embryoID}:`, err);
        }
    }

    console.log('All images retrieved.');

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