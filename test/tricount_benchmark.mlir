#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#row_sum_map_in = affine_map<(d0, d1) -> (d0, d1)>
#row_sum_map_out = affine_map<(d0, d1) -> (d0)>
#map_1d = affine_map<(d0) -> (d0)>

module {
    func.func private @start_timer()
    func.func private @stop_timer()

    func.func @tricount(%nodes_i32: i32, %edges_i32: i32, %coords_mem: memref<?x2xi64>, %vals_mem: memref<?xf32>, %out_tc: memref<?xf32>) attributes { llvm.emit_c_interface } {
        
        %num_nodes = arith.index_cast %nodes_i32 : i32 to index
        %num_edges = arith.index_cast %edges_i32 : i32 to index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %f0 = arith.constant 0.0 : f32
        %f2 = arith.constant 2.0 : f32

        // 1. Load Data
        %empty_coords = tensor.empty(%num_edges) : tensor<?x2xi64>
        %coords_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_coords = %empty_coords) -> tensor<?x2xi64> {
            %row = memref.load %coords_mem[%e, %c0] : memref<?x2xi64>
            %col = memref.load %coords_mem[%e, %c1] : memref<?x2xi64>
            %t1 = tensor.insert %row into %t_coords[%e, %c0] : tensor<?x2xi64>
            %t2 = tensor.insert %col into %t1[%e, %c1] : tensor<?x2xi64>
            scf.yield %t2 : tensor<?x2xi64>
        }

        %empty_vals = tensor.empty(%num_edges) : tensor<?xf32>
        %vals_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_vals = %empty_vals) -> tensor<?xf32> {
            %v = memref.load %vals_mem[%e] : memref<?xf32>
            %t1 = tensor.insert %v into %t_vals[%e] : tensor<?xf32>
            scf.yield %t1 : tensor<?xf32>
        }

        // 2. Build Adjacency Matrix
        %A = gblas.from_coo %coords_tensor, %vals_tensor (%num_nodes, %num_nodes)
            : tensor<?x2xi64>, tensor<?xf32> -> tensor<?x?xf32, #CSR>

        func.call @start_timer() : () -> ()

        // 3. C = (A * A) masked by A
        // Output is strictly sparse #CSR to prevent dense memory explosions
        %C = gblas.mxm %A, %A, %A
            combine = multiplies 
            reduce = plus 
            : tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR> 
            -> tensor<?x?xf32, #CSR>

        // 4. Reduce Rows to a 1D Vector (Row sums)
        %empty_1d = tensor.empty(%num_nodes) : tensor<?xf32>
        %zero_1d = linalg.fill ins(%f0 : f32) outs(%empty_1d : tensor<?xf32>) -> tensor<?xf32>

        %row_sums = linalg.generic {
            indexing_maps = [#row_sum_map_in, #row_sum_map_out],
            iterator_types = ["parallel", "reduction"]
        } ins(%C : tensor<?x?xf32, #CSR>) outs(%zero_1d : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %out : f32
            linalg.yield %sum : f32
        } -> tensor<?xf32>

        // 5. Divide each row sum by 2
        %tc_counts = linalg.generic {
            indexing_maps = [#map_1d, #map_1d],
            iterator_types = ["parallel"]
        } ins(%row_sums : tensor<?xf32>) outs(%empty_1d : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
            %tc_val = arith.divf %in, %f2 : f32
            linalg.yield %tc_val : f32
        } -> tensor<?xf32>

        func.call @stop_timer() : () -> ()

        // 6. Write out results
        scf.for %k = %c0 to %num_nodes step %c1 {
            %val = tensor.extract %tc_counts[%k] : tensor<?xf32>
            memref.store %val, %out_tc[%k] : memref<?xf32>
        }
        return
    }
}