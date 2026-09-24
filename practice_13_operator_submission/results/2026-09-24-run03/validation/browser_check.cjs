const {chromium}=require(process.env.P13_PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs'),assert=require('assert');
(async()=>{const browser=await chromium.launch({headless:true});try{
const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});const errors=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
await page.goto(require('url').pathToFileURL(require('path').resolve(process.argv[2] || require('path').join(__dirname,'../analysis/index.html'))).href);
assert.equal(await page.locator('#title').textContent(),'prefill · _triton_rope');
const count=await page.locator('#example option').count();assert(count>=32);
for(let i=0;i<count;i++){
 await page.locator('#example').selectOption(String(i));
 assert.equal(await page.locator('#timeline svg').count(),1);
 assert((await page.locator('#route').textContent()).includes('NPU stream 46'));
 assert((await page.locator('#times tr').count())>=6);
 assert(await page.locator('#parameters').textContent());
}
await page.locator('#task-search').fill('ReshapeAndCacheNdKernel');assert((await page.locator('#task-count').textContent()).includes('96条'));
await page.locator('#next').click();assert((await page.locator('#task-count').textContent()).includes('41–80'));
await page.locator('#previous').click();assert((await page.locator('#task-count').textContent()).includes('1–40'));
await page.locator('#task-search').fill('does-not-exist');assert((await page.locator('#task-count').textContent()).includes('0条'));
await page.locator('#task-search').fill('');assert((await page.locator('#task-count').textContent()).includes('1444条'));
assert.equal(await page.locator('#completion tr').count(),5);
for(const url of await page.locator('a').evaluateAll(as=>as.map(a=>a.href)))assert(fs.existsSync(new URL(url)),url);
const fia=await page.locator('#example option').evaluateAll(opts=>opts.findIndex(e=>e.textContent==='prefill · FusedInferAttentionScore'));
await page.locator('#example').selectOption(String(fia));
assert((await page.locator('#overlap').textContent()).includes('aten::view'));
await page.locator('#title').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/p13-detail.png'});
await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'/tmp/p13-top.png'});
await page.setViewportSize({width:390,height:844});
assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
await page.screenshot({path:'/tmp/p13-mobile.png'});
assert.deepEqual(errors,[]);assert(requests.every(x=>x.startsWith('file:')));
console.log(JSON.stringify({offline:true,operator_phase_examples_checked:count,pagination:true,search:true,completion_rows:4,links:true,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
