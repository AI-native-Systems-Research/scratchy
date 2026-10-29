//! Whole-run energy sampling for the Metal bench matrix.
//!
//! Reads Apple's IOReport (no root, nothing third-party) and prints one CSV
//! line per interval to stdout until SIGINT/SIGTERM:
//!
//!     unix_seconds,cpu_j,gpu_j,ane_j
//!
//! Modes:
//!   (no args)          sample and print CSV
//!   --interval-ms N    sampling interval (default 1000)
//!   --samples N        stop after N lines (default: until signalled)
//!   --dump             list every IOReport channel this machine exposes
//!   --dump-clpc        list the CLPC energy counters discovered (debug)
//!
//! Where the energy numbers come from: on Apple Silicon the energy counters
//! are CLPC reports (group "CLPC", subgroup "Energy Counters", cumulative
//! nanojoules). They are not in the global channel list, so each counter is
//! subscribed to as a synthesized channel description carrying the
//! AppleCLPC driver's registry-entry id and the counter's report id. The
//! report ids are per-chip-driver constants extracted from macOS 27.0 /
//! 27.0.1 (table below, discovered by the macmon project, MIT); on any other
//! macOS the sampler refuses to run rather than read garbage. Machines that
//! expose power or energy units in the global channel list (older chips)
//! are sampled from there as well, so the tool works across the fleet.
//!
//! The FFI declarations are written by hand: modern SDKs ship libIOReport as
//! a stub with no public header, and the calls below are the samples-based
//! API its symbol table actually exports (IOReportCreateSubscription ->
//! IOReportCreateSamples -> IOReportCreateSamplesDelta), verified against
//! the live tbd. Non-macOS builds compile to a loud error, not a silent
//! no-op.

#[cfg(target_os = "macos")]
mod ffi {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::ptr;

    pub type CFAllocatorRef = *const u8;
    pub type CFStringRef = *const u8;
    pub type CFTypeRef = *const u8;
    pub type CFNumberRef = *const u8;
    pub type CFDictionaryRef = *const u8;
    pub type CFMutableDictionaryRef = *mut u8;
    pub type CFArrayRef = *const u8;
    pub type CFMutableArrayRef = *mut u8;
    pub type CFIndex = isize;
    pub type Bool = c_int;
    pub type IOKitObject = u32;
    pub type IOKitIterator = u32;

    pub const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    pub const K_CF_NUMBER_S_INT_64: CFIndex = 4;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFStringCreateWithCString(
            alloc: CFAllocatorRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        pub fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: CFIndex,
            encoding: u32,
        ) -> Bool;
        pub fn CFNumberCreate(
            alloc: CFAllocatorRef,
            the_type: CFIndex,
            value: *const c_void,
        ) -> CFNumberRef;
        pub fn CFDictionaryGetValue(the_dict: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
        pub fn CFDictionaryGetCount(the_dict: CFDictionaryRef) -> CFIndex;
        pub fn CFDictionaryCreateMutable(
            alloc: CFAllocatorRef,
            capacity: CFIndex,
            key_callbacks: *const u8,
            value_callbacks: *const u8,
        ) -> CFMutableDictionaryRef;
        pub fn CFDictionaryCreateMutableCopy(
            alloc: CFAllocatorRef,
            capacity: CFIndex,
            dict: CFDictionaryRef,
        ) -> CFMutableDictionaryRef;
        pub fn CFDictionarySetValue(
            the_dict: CFMutableDictionaryRef,
            key: CFTypeRef,
            value: CFTypeRef,
        );
        pub fn CFArrayGetCount(array: CFArrayRef) -> CFIndex;
        pub fn CFArrayGetValueAtIndex(array: CFArrayRef, idx: CFIndex) -> CFTypeRef;
        pub fn CFArrayCreateMutable(
            alloc: CFAllocatorRef,
            capacity: CFIndex,
            callbacks: *const u8,
        ) -> CFMutableArrayRef;
        pub fn CFArrayAppendValue(the_array: CFMutableArrayRef, value: CFTypeRef);
        pub fn CFRelease(cf: CFTypeRef);
    }

    /// The CF container callback tables: with them, arrays and dictionaries
    /// retain what is put in, so owned temporaries can be dropped after
    /// appending. NULL callbacks (the default trap) retain nothing, and every
    /// appended pointer dangles the moment its owner is released.
    #[repr(C)]
    pub struct CFArrayCallBacks {
        pub version: CFIndex,
        pub retain: *const u8,
        pub release: *const u8,
        pub copy_description: *const u8,
        pub equal: *const u8,
        pub hash: *const u8,
    }

    #[repr(C)]
    pub struct CFDictionaryCallBacks {
        pub version: CFIndex,
        pub retain: *const u8,
        pub release: *const u8,
        pub copy_description: *const u8,
        pub equal: *const u8,
        pub hash: *const u8,
    }

    unsafe extern "C" {
        #[link_name = "kCFTypeArrayCallBacks"]
        pub static K_CF_TYPE_ARRAY_CALLBACKS: CFArrayCallBacks;
        #[link_name = "kCFTypeDictionaryKeyCallBacks"]
        pub static K_CF_TYPE_DICTIONARY_KEY_CALLBACKS: CFDictionaryCallBacks;
        #[link_name = "kCFTypeDictionaryValueCallBacks"]
        pub static K_CF_TYPE_DICTIONARY_VALUE_CALLBACKS: CFDictionaryCallBacks;
    }

    #[link(name = "IOReport")]
    unsafe extern "C" {
        // The samples-based API libIOReport.tbd exports.
        pub fn IOReportCopyAllChannels(a: u64, b: u64) -> CFDictionaryRef;
        pub fn IOReportCopyChannelsInGroup(
            group: CFStringRef,
            subgroup: CFStringRef,
            a: u64,
            b: u64,
            c: u64,
        ) -> CFDictionaryRef;
        pub fn IOReportCreateSubscription(
            a: *const u8,
            channels: CFMutableDictionaryRef,
            subscribed: *mut CFMutableDictionaryRef,
            options: u64,
            context: *const u8,
        ) -> *mut u8;
        pub fn IOReportCreateSamples(
            subscription: *mut u8,
            channels: CFMutableDictionaryRef,
            context: *const u8,
        ) -> CFDictionaryRef;
        pub fn IOReportCreateSamplesDelta(
            a: CFDictionaryRef,
            b: CFDictionaryRef,
            context: *const u8,
        ) -> CFDictionaryRef;
        pub fn IOReportChannelGetGroup(channel: CFDictionaryRef) -> CFStringRef;
        pub fn IOReportChannelGetSubGroup(channel: CFDictionaryRef) -> CFStringRef;
        pub fn IOReportChannelGetChannelName(channel: CFDictionaryRef) -> CFStringRef;
        pub fn IOReportChannelGetUnitLabel(channel: CFDictionaryRef) -> CFStringRef;
        pub fn IOReportSimpleGetIntegerValue(channel: CFDictionaryRef, index: i32) -> i64;
    }

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        // The AppleCLPC services whose registry-entry ids the synthesized
        // CLPC channel descriptions carry.
        pub fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
        pub fn IOServiceGetMatchingServices(
            main_port: u32, // 0: kIOMainPortDefault
            matching: CFMutableDictionaryRef,
            existing: *mut IOKitIterator,
        ) -> c_int;
        pub fn IOIteratorNext(iterator: IOKitIterator) -> IOKitObject;
        pub fn IOObjectRelease(object: IOKitObject) -> c_int;
        pub fn IORegistryEntryGetRegistryEntryID(entry: IOKitObject, id: *mut u64) -> c_int;
        pub fn IORegistryEntryCreateCFProperties(
            entry: IOKitObject,
            properties: *mut CFMutableDictionaryRef,
            allocator: CFAllocatorRef,
            options: u32,
        ) -> c_int;
    }

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// A CF object we own, released on drop.
    pub struct Owned(pub CFTypeRef);
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe { CFRelease(self.0) }
        }
    }

    /// A mutable CF container we own, released on drop.
    pub struct OwnedMut(pub *mut u8);
    impl Drop for OwnedMut {
        fn drop(&mut self) {
            unsafe { CFRelease(self.0 as CFTypeRef) }
        }
    }

    pub fn cfstr(s: &str) -> Owned {
        let c = CString::new(s).expect("no interior NUL");
        Owned(unsafe {
            CFStringCreateWithCString(ptr::null(), c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
        })
    }

    pub fn cfnum(v: i64) -> Owned {
        Owned(unsafe {
            CFNumberCreate(
                ptr::null(),
                K_CF_NUMBER_S_INT_64,
                &v as *const i64 as *const c_void,
            )
        })
    }

    pub fn dict_new() -> OwnedMut {
        OwnedMut(unsafe {
            CFDictionaryCreateMutable(
                ptr::null(),
                0,
                &K_CF_TYPE_DICTIONARY_KEY_CALLBACKS as *const _ as *const u8,
                &K_CF_TYPE_DICTIONARY_VALUE_CALLBACKS as *const _ as *const u8,
            )
        })
    }

    pub fn array_new() -> OwnedMut {
        OwnedMut(unsafe {
            CFArrayCreateMutable(
                ptr::null(),
                0,
                &K_CF_TYPE_ARRAY_CALLBACKS as *const _ as *const u8,
            )
        })
    }

    /// Copy a CFStringRef into a Rust String (truncates past 255 bytes).
    /// NULL-safe: sample items can omit metadata fields entirely.
    pub fn to_rust(s: CFStringRef) -> String {
        if s.is_null() {
            return String::new();
        }
        let mut buf = [0 as c_char; 256];
        if unsafe {
            CFStringGetCString(
                s,
                buf.as_mut_ptr(),
                buf.len() as CFIndex,
                K_CF_STRING_ENCODING_UTF8,
            )
        } == 0
        {
            return String::new();
        }
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// kern.osproductversion, e.g. "27.0.1".
    pub fn os_product_version() -> String {
        let name = CString::new("kern.osproductversion").unwrap();
        let mut buf = [0u8; 32];
        let mut len = buf.len();
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
                ptr::null(),
                0,
            )
        };
        if rc != 0 {
            return String::new();
        }
        String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).into_owned()
    }
}

#[cfg(target_os = "macos")]
mod clpc {
    use super::ffi;

    /// The aggregate CLPC energy counter keys, per chip driver, and the
    /// report indices that carry them on macOS 27. Discovered from
    /// 27.0 (26A428) and 27.0.1 (26A434) — table published by the macmon
    /// project (MIT). A driver not listed here has no known counters; the
    /// sampler says so rather than guessing ids.
    struct Keys {
        bundle: &'static str,
        cpu: u32,
        gpu: u32,
        ane: u32,
    }

    const INDICES_27: (u32, u32, u32) = (16, 25, 24); // (cpu, gpu, ane)

    const KEYS: &[Keys] = &[
        Keys {
            bundle: "com.apple.driver.AppleT6000CLPCv3",
            cpu: 0x47479059,
            gpu: 0x4519bf1e,
            ane: 0x9aaa17f8,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6002CLPC",
            cpu: 0xb501816b,
            gpu: 0x9e2c3e8b,
            ane: 0x14d5a574,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6020CLPC",
            cpu: 0xe301cd75,
            gpu: 0x8f34cc7a,
            ane: 0x7415fb0f,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6022CLPC",
            cpu: 0x76f5fd21,
            gpu: 0x90624499,
            ane: 0x3c097307,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6030CLPC",
            cpu: 0xbe03a01a,
            gpu: 0x472d6c9c,
            ane: 0xb8990c38,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6031CLPC",
            cpu: 0xd81e0084,
            gpu: 0x1085f98b,
            ane: 0x313a3be1,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6032CLPC",
            cpu: 0xf4a21308,
            gpu: 0x85e80513,
            ane: 0x63dc8e25,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6041CLPC",
            cpu: 0x263cf1f0,
            gpu: 0x22e4cf59,
            ane: 0x7849fb5c,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6050CLPC",
            cpu: 0xd3ab60bb,
            gpu: 0x27cd4ce5,
            ane: 0x8c0f1c59,
        },
        Keys {
            bundle: "com.apple.driver.AppleT6050dCLPC",
            cpu: 0x18b3010e,
            gpu: 0xedeca11a,
            ane: 0x188bd883,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8103CLPCv3",
            cpu: 0x2cdd31f2,
            gpu: 0x21becf42,
            ane: 0x0667fb81,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8112CLPC",
            cpu: 0xb5435137,
            gpu: 0x638d9d52,
            ane: 0xea089c36,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8122CLPC",
            cpu: 0x54b51c99,
            gpu: 0x0ce2d9dc,
            ane: 0x960afe2f,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8132CLPC",
            cpu: 0x4a854d94,
            gpu: 0x3b3e3b79,
            ane: 0x1ff4a9ec,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8140CLPC",
            cpu: 0x9bf54436,
            gpu: 0xef70589c,
            ane: 0x569d941b,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8142CLPC",
            cpu: 0x43708744,
            gpu: 0x6a010933,
            ane: 0xbc5efdf0,
        },
        Keys {
            bundle: "com.apple.driver.AppleT8152CLPC",
            cpu: 0x485d4bfc,
            gpu: 0x11784654,
            ane: 0x9380c6bb,
        },
    ];

    /// One AppleCLPC service found in the IORegistry, with the counters its
    /// driver exposes on this macOS. `channels` is empty for a driver the
    /// table does not cover.
    pub struct Service {
        pub driver_id: u64,
        pub bundle: String,
        /// (report id, rail) per counter: rail is "cpu", "gpu" or "ane".
        pub channels: Vec<(u64, &'static str)>,
    }

    /// Whether this macOS reads energy from the CLPC counters: on 27, where
    /// "Energy Model"'s CPU Energy channel reads 0 under load. Every other
    /// release reads the "Energy Model" group by channel name.
    pub fn is_clpc_release(version: &str) -> bool {
        version.starts_with("27.")
    }

    /// Find every AppleCLPC service and the counters it carries.
    pub fn discover() -> Vec<Service> {
        let (cpu_i, gpu_i, ane_i) = INDICES_27;
        let mut out = Vec::new();
        let name = std::ffi::CString::new("AppleCLPC").unwrap();
        let mut iter: ffi::IOKitIterator = 0;
        let matching = unsafe { ffi::IOServiceMatching(name.as_ptr()) };
        let kr = unsafe { ffi::IOServiceGetMatchingServices(0, matching, &mut iter) };
        if kr != 0 {
            return out;
        }
        loop {
            let entry = unsafe { ffi::IOIteratorNext(iter) };
            if entry == 0 {
                break;
            }
            let mut driver_id = 0u64;
            let mut props: ffi::CFMutableDictionaryRef = std::ptr::null_mut();
            let ok = unsafe { ffi::IORegistryEntryGetRegistryEntryID(entry, &mut driver_id) } == 0
                && unsafe {
                    ffi::IORegistryEntryCreateCFProperties(entry, &mut props, std::ptr::null(), 0)
                } == 0;
            unsafe { ffi::IOObjectRelease(entry) };
            if !ok {
                continue;
            }
            let props = ffi::OwnedMut(props);
            let key = ffi::cfstr("CFBundleIdentifier");
            let bundle = unsafe { ffi::CFDictionaryGetValue(props.0, key.0) };
            let bundle = if bundle.is_null() {
                String::new()
            } else {
                ffi::to_rust(bundle as ffi::CFStringRef)
            };
            let channels = KEYS
                .iter()
                .find(|k| k.bundle == bundle)
                .map(|k| {
                    vec![
                        (((cpu_i as u64) << 32) | k.cpu as u64, "cpu"),
                        (((gpu_i as u64) << 32) | k.gpu as u64, "gpu"),
                        (((ane_i as u64) << 32) | k.ane as u64, "ane"),
                    ]
                })
                .unwrap_or_default();
            out.push(Service {
                driver_id,
                bundle,
                channels,
            });
        }
        out
    }

    /// Build the synthesized channel description for one counter: the dict
    /// IOReport subscribes to, carrying the driver id, the report id and the
    /// nJ unit. Layout per IOReportTypes.h: simple integer channel, power
    /// category, one element.
    pub fn channel_description(driver_id: u64, report_id: u64, name: &str) -> ffi::OwnedMut {
        let channel_type: i64 = 1 | (2 << 16) | (1 << 32);
        let unit: i64 = (3 << 56) | (118 << 32);

        let legend = ffi::array_new();
        let id = ffi::cfnum(report_id as i64);
        let ty = ffi::cfnum(channel_type);
        let nm = ffi::cfstr(name);
        unsafe {
            ffi::CFArrayAppendValue(legend.0, id.0);
            ffi::CFArrayAppendValue(legend.0, ty.0);
            ffi::CFArrayAppendValue(legend.0, nm.0);
        }

        let info = ffi::dict_new();
        let uk = ffi::cfstr("IOReportChannelUnit");
        let uv = ffi::cfnum(unit);
        unsafe { ffi::CFDictionarySetValue(info.0, uk.0, uv.0) };

        let dict = ffi::dict_new();
        // Every value is a named local: with .0 copied out of an unnamed
        // temporary, the temporary's Drop would free the CF object before
        // the dict retains it.
        let v_driver = ffi::cfnum(driver_id as i64);
        let n_apple = ffi::cfstr("AppleCLPC");
        let n_group = ffi::cfstr("CLPC");
        let n_sub = ffi::cfstr("Energy Counters");
        let fields = [
            (ffi::cfstr("DriverID"), v_driver.0),
            (ffi::cfstr("DriverName"), n_apple.0),
            (ffi::cfstr("IOReportGroupName"), n_group.0),
            (ffi::cfstr("IOReportSubGroupName"), n_sub.0),
            (ffi::cfstr("IOReportChannelInfo"), info.0 as ffi::CFTypeRef),
            (ffi::cfstr("LegendChannel"), legend.0 as ffi::CFTypeRef),
        ];
        for (k, v) in fields {
            unsafe { ffi::CFDictionarySetValue(dict.0, k.0, v) };
        }
        dict
    }
}

#[cfg(target_os = "macos")]
#[derive(PartialEq)]
enum Mode {
    Dump,
    DumpClpc,
    Csv,
}

#[cfg(target_os = "macos")]
fn main() {
    let mut interval_ms = 1000u64;
    let mut samples: Option<u64> = None;
    let mut mode = Mode::Csv;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump" => mode = Mode::Dump,
            "--dump-clpc" => mode = Mode::DumpClpc,
            "--samples" => {
                i += 1;
                samples = args.get(i).and_then(|v| v.parse().ok());
                if samples.is_none() {
                    eprintln!("--samples takes a count");
                    std::process::exit(2);
                }
            }
            "--interval-ms" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    interval_ms = v.parse().unwrap_or(1000);
                }
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let version = ffi::os_product_version();
    let clpc_release = clpc::is_clpc_release(&version);
    let services = if clpc_release {
        clpc::discover()
    } else {
        Vec::new()
    };
    eprintln!(
        "macOS {version}: energy from {}",
        if clpc_release {
            "the CLPC counters"
        } else {
            "IOReport's Energy Model group"
        }
    );

    if mode == Mode::DumpClpc {
        if services.is_empty() {
            eprintln!("no AppleCLPC service found");
        }
        for s in &services {
            eprintln!("AppleCLPC driver {:#x} ({})", s.driver_id, s.bundle);
            if s.channels.is_empty() {
                eprintln!("  no counter ids known for this driver");
            }
            for (id, rail) in &s.channels {
                eprintln!("  {rail} energy report {id:#018x}");
            }
        }
        return;
    }

    let all = unsafe { ffi::IOReportCopyAllChannels(0, 0) };
    if all.is_null() {
        eprintln!("IOReportCopyAllChannels failed");
        std::process::exit(1);
    }
    let channels_key = ffi::cfstr("IOReportChannels");
    let all_arr = unsafe { ffi::CFDictionaryGetValue(all, channels_key.0) } as ffi::CFArrayRef;

    if mode == Mode::Dump {
        let n = unsafe { ffi::CFArrayGetCount(all_arr) };
        eprintln!("{n} channels");
        for i in 0..n {
            let ch = unsafe { ffi::CFArrayGetValueAtIndex(all_arr, i) } as ffi::CFDictionaryRef;
            println!(
                "{} | {} | {} | {}",
                ffi::to_rust(unsafe { ffi::IOReportChannelGetGroup(ch) }),
                ffi::to_rust(unsafe { ffi::IOReportChannelGetSubGroup(ch) }),
                ffi::to_rust(unsafe { ffi::IOReportChannelGetChannelName(ch) }),
                ffi::to_rust(unsafe { ffi::IOReportChannelGetUnitLabel(ch) }),
            );
        }
        return;
    }

    sample_csv(
        all,
        &channels_key,
        &services,
        clpc_release,
        interval_ms,
        samples,
    );
}

#[cfg(target_os = "macos")]
fn sample_csv(
    all: ffi::CFDictionaryRef,
    channels_key: &ffi::Owned,
    services: &[clpc::Service],
    clpc_release: bool,
    interval_ms: u64,
    samples: Option<u64>,
) {
    use std::thread::sleep;
    use std::time::Duration;

    // The selection: the synthesized CLPC counters on a CLPC release, else
    // the "Energy Model" group's channels, read by name.
    let selected = ffi::array_new();
    if !clpc_release {
        let group = ffi::cfstr(ENERGY_MODEL);
        let em = unsafe { ffi::IOReportCopyChannelsInGroup(group.0, std::ptr::null(), 0, 0, 0) };
        if em.is_null() {
            eprintln!("error: IOReport has no {ENERGY_MODEL:?} group on this machine");
            std::process::exit(1);
        }
        let em = ffi::Owned(em);
        let arr = unsafe { ffi::CFDictionaryGetValue(em.0, channels_key.0) } as ffi::CFArrayRef;
        let n = if arr.is_null() {
            0
        } else {
            unsafe { ffi::CFArrayGetCount(arr) }
        };
        for i in 0..n {
            let ch = unsafe { ffi::CFArrayGetValueAtIndex(arr, i) };
            unsafe { ffi::CFArrayAppendValue(selected.0, ch) };
        }
    }
    let clpc_wanted: Vec<&'static str> = services
        .iter()
        .flat_map(|s| s.channels.iter().map(|&(_, rail)| rail))
        .collect();
    for s in services {
        for &(id, rail) in &s.channels {
            let desc = clpc::channel_description(s.driver_id, id, rail);
            unsafe { ffi::CFArrayAppendValue(selected.0, desc.0 as ffi::CFTypeRef) };
        }
    }
    let selected_n = unsafe { ffi::CFArrayGetCount(selected.0) };
    eprintln!(
        "{selected_n} energy channels selected ({} CLPC counters)",
        clpc_wanted.len()
    );
    if selected_n == 0 {
        std::process::exit(1);
    }

    // The subscription takes a copy of the all-channels dict whose
    // IOReportChannels holds our selection; the dict it returns is the one
    // to sample.
    let cap = unsafe { ffi::CFDictionaryGetCount(all) };
    let chan = unsafe { ffi::CFDictionaryCreateMutableCopy(std::ptr::null(), cap, all) };
    unsafe { ffi::CFDictionarySetValue(chan, channels_key.0, selected.0 as ffi::CFTypeRef) };
    let mut subscribed: ffi::CFMutableDictionaryRef = std::ptr::null_mut();
    let subs = unsafe {
        ffi::IOReportCreateSubscription(
            std::ptr::null(),
            chan,
            &mut subscribed,
            0,
            std::ptr::null(),
        )
    };
    if subs.is_null() || subscribed.is_null() {
        eprintln!("IOReportCreateSubscription failed");
        std::process::exit(1);
    }

    let mut prev = unsafe { ffi::IOReportCreateSamples(subs, subscribed, std::ptr::null()) };
    if prev.is_null() {
        eprintln!("IOReportCreateSamples failed");
        std::process::exit(1);
    }
    let mut checked = false;
    let mut written = 0u64;
    while samples.is_none_or(|n| written < n) {
        sleep(Duration::from_millis(interval_ms));
        let next = unsafe { ffi::IOReportCreateSamples(subs, subscribed, std::ptr::null()) };
        let delta = unsafe { ffi::IOReportCreateSamplesDelta(prev, next, std::ptr::null()) };
        prev = next;

        // One CSV line: unix_seconds,cpu_j,gpu_j,ane_j. CLPC counters are
        // nJ over the interval; global power channels are mW, integrated
        // over the interval. Rail by channel name.
        let mut clpc = [0.0f64; 3];
        let mut global = [0.0f64; 3];
        let mut saw = [false; 3];
        let mut saw_global = [false; 3];
        let arr = unsafe { ffi::CFDictionaryGetValue(delta, channels_key.0) } as ffi::CFArrayRef;
        if arr.is_null() {
            continue; // a delta with no channel array: nothing to read
        }
        let dn = unsafe { ffi::CFArrayGetCount(arr) };
        for i in 0..dn {
            let ch = unsafe { ffi::CFArrayGetValueAtIndex(arr, i) } as ffi::CFDictionaryRef;
            let group = ffi::to_rust(unsafe { ffi::IOReportChannelGetGroup(ch) });
            let name = ffi::to_rust(unsafe { ffi::IOReportChannelGetChannelName(ch) });
            let unit = ffi::to_rust(unsafe { ffi::IOReportChannelGetUnitLabel(ch) })
                .trim()
                .to_string();
            let v = unsafe { ffi::IOReportSimpleGetIntegerValue(ch, 0) } as f64;
            if group == "CLPC" {
                // Our synthesized counters: the channel name is the rail.
                let idx = match name.as_str() {
                    "cpu" => 0,
                    "gpu" => 1,
                    "ane" => 2,
                    _ => continue,
                };
                saw[idx] = true;
                clpc[idx] += v / 1e9;
            } else if group == ENERGY_MODEL {
                let Some(idx) = ENERGY_MODEL_RAILS.iter().position(|&r| r == name) else {
                    continue;
                };
                let Some(per_joule) = joules_per_unit(&unit) else {
                    eprintln!("error: {ENERGY_MODEL}/{name} is in {unit:?}, not mJ, uJ or nJ");
                    std::process::exit(1);
                };
                saw_global[idx] = true;
                global[idx] += v * per_joule;
            }
        }
        // Where the CLPC counters are live they are the number; global
        // channels (which on some chips duplicate them as estimates) do not
        // add to them.
        let out = [
            if saw[0] { clpc[0] } else { global[0] },
            if saw[1] { clpc[1] } else { global[1] },
            if saw[2] { clpc[2] } else { global[2] },
        ];
        // Every rail needs a live source on the first interval: a rail with
        // none would be written as 0 J, so the run stops instead.
        if !checked {
            let missing: Vec<&str> = ["cpu", "gpu", "ane"]
                .into_iter()
                .zip(saw.iter().zip(saw_global))
                .filter(|&(_, (&c, g))| !c && !g)
                .map(|(rail, _)| rail)
                .collect();
            if !missing.is_empty() {
                eprintln!("error: no energy channel for: {}", missing.join(", "));
                std::process::exit(1);
            }
            checked = true;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        println!("{now:.3},{:.3},{:.3},{:.3}", out[0], out[1], out[2]);
        written += 1;
    }
}

/// The IOReport group every release before 27 reads energy from, by name.
#[cfg(target_os = "macos")]
const ENERGY_MODEL: &str = "Energy Model";

/// Its channels for cpu, gpu and ane, in that order.
#[cfg(target_os = "macos")]
const ENERGY_MODEL_RAILS: [&str; 3] = ["CPU Energy", "GPU Energy", "ANE"];

/// Joules per count of an IOReport energy unit label.
#[cfg(any(target_os = "macos", test))]
fn joules_per_unit(unit: &str) -> Option<f64> {
    match unit {
        "mJ" => Some(1e-3),
        "uJ" | "µJ" => Some(1e-6),
        "nJ" => Some(1e-9),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_units_convert_to_joules() {
        assert_eq!(joules_per_unit("mJ"), Some(1e-3));
        assert_eq!(joules_per_unit("uJ"), Some(1e-6));
        assert_eq!(joules_per_unit("nJ"), Some(1e-9));
        assert_eq!(joules_per_unit("mW"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn only_macos_27_reads_clpc() {
        assert!(clpc::is_clpc_release("27.0.1"));
        assert!(clpc::is_clpc_release("27.2"));
        assert!(!clpc::is_clpc_release("26.4"));
        assert!(!clpc::is_clpc_release("28.0"));
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("scratchy-power-sampler is macOS-only: it reads Apple's IOReport");
    std::process::exit(1);
}
