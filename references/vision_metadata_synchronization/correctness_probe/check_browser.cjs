// Supply PLAYWRIGHT_MODULE when Playwright is installed outside this project.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
(async()=>{
 const root=__dirname, data=JSON.parse(fs.readFileSync(path.join(root,'results/run-r01/cases.json'),'utf8'));
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()));
 await page.goto(pathToFileURL(path.join(root,'report/index.html')).href);
 if(await page.locator('#case').inputValue()!=='same_sum')throw Error('wrong default');
 for(const r of data){
  await page.locator('#case').selectOption(r.case.id);
  if(await page.locator('#case-title').textContent()!==r.case.label)throw Error('wrong scenario');
  if(await page.locator('#checks .fail').count()!==Object.values(r.checks).filter(x=>!x).length)throw Error('wrong failures');
  if(await page.locator('#memory .cell').count()!==r.parent_before.length)throw Error('wrong memory count');
  if(!(await page.locator('#split-detail').textContent()).includes(r.split.status))throw Error('wrong split result');
 }
 await page.locator('#case').selectOption('offset');
 if(await page.locator('#memory .left').count()!==3||await page.locator('#memory .right').count()!==3)throw Error('wrong slice highlight');
 await page.locator('#matrix button').filter({hasText:'总和不变的错误分段'}).click();
 if(await page.locator('#case').inputValue()!=='same_sum')throw Error('matrix jump failed');
 await page.screenshot({path:'/tmp/lengths-correctness-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 for(const r of data){
  await page.locator('#case').selectOption(r.case.id);
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow '+r.case.id);
 }
 await page.locator('#case').selectOption('stride2');
 await page.screenshot({path:'/tmp/lengths-correctness-mobile.png',fullPage:true});
 if(errors.length)throw Error(errors.join('\n'));
 const result={status:'passed',offline:true,scenario_switches:data.length,mobile_scenarios:data.length,
     memory_highlights:true,matrix_jump:true,mobile_overflow:false,page_errors:errors};
 fs.writeFileSync(path.join(root,'results/browser.json'),JSON.stringify(result,null,2)+'\n');
 console.log(result);await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
