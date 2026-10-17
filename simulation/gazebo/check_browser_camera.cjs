// Requires the running local console/Gazebo, Playwright and Chromium.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH || undefined,headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:1180}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const status=async()=>await(await page.request.get('http://127.0.0.1:8767/api/status')).json();
 const wait=async predicate=>{const end=Date.now()+30000;while(Date.now()<end){let s=await status();if(predicate(s))return s;await page.waitForTimeout(75);}throw new Error('Camera/IMU timeout');};
 const action=async name=>{const response=await page.request.post('http://127.0.0.1:8767/api/action',{data:{action:name}});assert(response.ok(),await response.text());};
 try{
  await page.goto('http://127.0.0.1:8767/');await wait(s=>s.online&&s.camera.streaming);
  await action('reset');await action('play');
  await wait(s=>s.camera.streaming&&s.imu.heading_calibrated);
  await page.waitForFunction(()=>document.querySelector('#camera-image').naturalWidth===1280);
  const response=await page.request.get('http://127.0.0.1:8767/camera.png'),png=await response.body();
  assert(response.ok());assert.equal(png.subarray(1,4).toString(),'PNG');assert.equal(png[25],0,'Preview is not grayscale');
  assert.equal(png.readUInt32BE(16),1280);assert.equal(png.readUInt32BE(20),720);
  const before=await status();await page.keyboard.down('q');await page.waitForTimeout(700);await page.keyboard.up('q');
  await wait(s=>!s.driving&&s.camera.sequence>before.camera.sequence+5);
  await page.waitForTimeout(600);const after=await status();
  assert(Math.abs(after.pose.yaw-before.pose.yaw)>.8,'Camera is not attached to a turning car');
  assert.equal(await page.locator('#camera-status').textContent(),'LIVE');
  assert.deepEqual(errors,[]);
  if(process.env.CAMERA_SCREENSHOT)await page.screenshot({path:process.env.CAMERA_SCREENSHOT,fullPage:true});
  console.log(JSON.stringify({image:[1280,720],format:'grayscale PNG',camera_streaming:true,keyboard_pivot:true,heading_error_deg:after.imu.heading_error_deg}));
  console.log('PASS: live mono preview, image dimensions, keyboard pivot and IMU coexistence');
 } finally {await action('reset');await action('play');await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
