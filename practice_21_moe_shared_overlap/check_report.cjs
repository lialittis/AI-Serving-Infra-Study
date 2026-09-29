const {chromium}=require(process.env.P21_BROWSER_PACKAGE||'playwright');
const assert=require('assert'),{pathToFileURL}=require('url'),path=require('path');
(async()=>{const browser=await chromium.launch({headless:true});try{
const page=await browser.newPage({viewport:{width:1400,height:1000},offline:true});
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
await page.goto(pathToFileURL(path.resolve(__dirname,'report/index.html')).href);
let trials=0;
for(const n of ['1','32','256','1024'])for(const mode of ['serial','parallel'])for(const r of ['0','1','2']){
 await page.locator('#tokens').selectOption(n);await page.locator('#mode').selectOption(mode);await page.locator('#repeat').selectOption(r);
 assert.equal(await page.locator('[data-stage]').count(),6);
 await page.locator('[data-stage="shared_down"]').click();let details=JSON.parse(await page.locator('#details').textContent());assert.equal(details.kernels.length,5);assert(details.calls.length);
 const tasks=page.locator('[data-task]');assert(await tasks.count()>16);
 const index=await tasks.evaluateAll(xs=>xs.findIndex(e=>e.tagName==='rect'&&Number(e.getAttribute('width'))>2));assert(index>=0);
 await tasks.nth(index).click();details=JSON.parse(await page.locator('#details').textContent());assert(details.node.cann_flow);assert(details.call);
 trials++;
}
for(const kind of ['event_wait','data_dependency','stream_order',''])await page.locator('#edges').selectOption(kind);
await page.locator('#edges').selectOption('event_wait');await page.locator('#zoom').selectOption('5');await page.locator('#zoom').selectOption('1');
await page.screenshot({path:'/tmp/p21-desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});
assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:'/tmp/p21-mobile.png',fullPage:true});
assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
console.log(JSON.stringify({offline:true,trials,stage_and_kernel_details:true,edge_filters:4,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
