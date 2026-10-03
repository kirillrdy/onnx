//! Minimal Metal compute driver used by the native graph runtime. The public
//! surface deliberately matches the CUDA and OpenCL packages.

const std = @import("std");

pub const is_metal = true;
pub const Error = error{Metal};

const c = @import("c");

pub const DevicePtr = extern struct {
    buffer: ?*anyopaque,
    offset: usize,
};
pub const null_ptr: DevicePtr = .{ .buffer = null, .offset = 0 };
pub const Dim = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };

threadlocal var current_context: ?Context = null;

pub fn lastError() []const u8 {
    return std.mem.span(c.sam_metal_last_error());
}

pub fn init() Error!void {}

pub const Context = struct {
    ptr: *c.SamMetalContext,

    pub fn init(ordinal: u32) Error!Context {
        const result: Context = .{ .ptr = c.sam_metal_context_create(ordinal) orelse return Error.Metal };
        current_context = result;
        return result;
    }

    pub fn deinit(self: Context) void {
        c.sam_metal_context_destroy(self.ptr);
        if (current_context != null and current_context.?.ptr == self.ptr) current_context = null;
    }

    pub fn makeCurrent(self: Context) Error!void {
        current_context = self;
    }
    pub fn synchronize(self: Context) Error!void {
        if (c.sam_metal_context_synchronize(self.ptr) == 0) return Error.Metal;
    }
};

pub const Module = struct {
    ptr: *c.SamMetalModule,
    context: Context,

    pub fn load(source: [:0]const u8) Error!Module {
        const context = current_context orelse return Error.Metal;
        return .{ .ptr = c.sam_metal_module_create(context.ptr, source.ptr) orelse return Error.Metal, .context = context };
    }

    pub fn unload(self: Module) void {
        c.sam_metal_module_destroy(self.ptr);
    }

    pub fn function(self: Module, name: []const u8) Error!Function {
        var symbol: [256:0]u8 = undefined;
        if (name.len >= symbol.len) return Error.Metal;
        @memcpy(symbol[0..name.len], name);
        symbol[name.len] = 0;
        return .{ .ptr = c.sam_metal_function_create(self.ptr, &symbol) orelse return Error.Metal };
    }
};

pub const Function = struct {
    ptr: *c.SamMetalFunction,

    pub fn launch(self: Function, grid: Dim, block: Dim, args: anytype) Error!void {
        if (grid.x == 0 or grid.y == 0 or grid.z == 0) return;
        if (block.x == 0 or block.y == 0 or block.z == 0) return;
        const field_names = comptime std.meta.fieldNames(@TypeOf(args));
        comptime var types: [field_names.len]type = undefined;
        inline for (field_names, 0..) |name, i| types[i] = @TypeOf(@field(args, name));
        var values: @Tuple(&types) = args;
        var encoded_args: [field_names.len]c.SamMetalArg = undefined;
        inline for (field_names, 0..) |name, i| {
            if (@TypeOf(@field(args, name)) == DevicePtr) {
                encoded_args[i] = .{ .kind = c.SAM_METAL_BUFFER, .buffer = bufferRef(@field(values, name)), .bytes = null, .size = 0 };
            } else {
                encoded_args[i] = .{ .kind = c.SAM_METAL_BYTES, .buffer = bufferRef(null_ptr), .bytes = &@field(values, name), .size = @sizeOf(@TypeOf(@field(args, name))) };
            }
        }
        const metal_grid: c.SamMetalDim = .{ .x = grid.x, .y = grid.y, .z = grid.z };
        const metal_block: c.SamMetalDim = .{ .x = block.x, .y = block.y, .z = block.z };
        if (c.sam_metal_launch(self.ptr, metal_grid, metal_block, &encoded_args, encoded_args.len) == 0) return Error.Metal;
    }
};

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();
        ptr: DevicePtr,
        len: usize,

        pub fn alloc(len: usize) Error!Self {
            if (len == 0) return .{ .ptr = null_ptr, .len = 0 };
            const context = current_context orelse return Error.Metal;
            return .{ .ptr = .{ .buffer = c.sam_metal_buffer_create(context.ptr, len * @sizeOf(T)) orelse return Error.Metal, .offset = 0 }, .len = len };
        }

        pub fn free(self: Self) void {
            c.sam_metal_buffer_destroy(self.ptr.buffer);
        }

        pub fn slice(self: Self, offset: usize, len: usize) Self {
            std.debug.assert(offset + len <= self.len);
            return .{ .ptr = .{ .buffer = self.ptr.buffer, .offset = self.ptr.offset + offset * @sizeOf(T) }, .len = len };
        }

        pub fn upload(self: Self, host: []const T) Error!void {
            return self.uploadAt(0, host);
        }

        pub fn uploadAt(self: Self, offset: usize, host: []const T) Error!void {
            std.debug.assert(offset + host.len <= self.len);
            if (host.len == 0) return;
            // A direct shared-memory write must not overtake kernels already
            // reading this buffer. Synchronous uploads are uncommon graph
            // boundaries, so retire the queue before exposing the write.
            if (current_context) |context| try context.synchronize();
            const destination: DevicePtr = .{
                .buffer = self.ptr.buffer,
                .offset = self.ptr.offset + offset * @sizeOf(T),
            };
            c.sam_metal_buffer_upload(bufferRef(destination), host.ptr, host.len * @sizeOf(T));
        }
        pub fn uploadAsync(self: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (host.len == 0) return;
            const context = current_context orelse return Error.Metal;
            if (c.sam_metal_buffer_upload_async(context.ptr, bufferRef(self.ptr), host.ptr, host.len * @sizeOf(T)) == 0)
                return Error.Metal;
        }
        pub fn download(self: Self, host: []T) Error!void {
            std.debug.assert(host.len <= self.len);
            if (current_context) |context| try context.synchronize();
            if (host.len != 0) c.sam_metal_buffer_download(bufferRef(self.ptr), host.ptr, host.len * @sizeOf(T));
        }

        pub fn copy(self: Self, src: Self) Error!void {
            std.debug.assert(src.len <= self.len);
            if (src.len == 0 or self.ptr.buffer == null or src.ptr.buffer == null) return;
            const context = current_context orelse return Error.Metal;
            if (c.sam_metal_buffer_copy(context.ptr, bufferRef(self.ptr), bufferRef(src.ptr), src.len * @sizeOf(T)) == 0)
                return Error.Metal;
        }
    };
}

// Convert the shared ABI fields explicitly: Zig 0.17 disallows extern-struct bitcasts.
fn bufferRef(ptr: DevicePtr) c.SamMetalBufferRef {
    return .{ .buffer = ptr.buffer, .offset = ptr.offset };
}
